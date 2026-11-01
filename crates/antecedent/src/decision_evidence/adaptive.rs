//! Compiler-owned adaptive-policy evidence for decision composition.
// SPDX-License-Identifier: MIT OR Apache-2.0
use super::*;
use crate::analysis::joint_panel::designs::{NativeJointDesignResult, PreparedNativeJointDesign};
use antecedent_estimate::joint_design::compiled::{
    JointDesignInference, JointDesignRecipe, JointNativeDesignLaw,
};

/// Adaptive-policy evidence with its retained executable source and admission lease.
#[derive(Clone, Debug)]
pub struct CompiledAdaptiveRegimeEvidence {
    source: NativeJointDesignResult,
    evidence: JointExperimentEvidence,
}
impl CompiledAdaptiveRegimeEvidence {
    /// Exact compiler source and dependency identity.
    #[must_use]
    pub fn source(&self) -> &NativeJointDesignResult {
        &self.source
    }
    /// Historical numerical evidence; current use still requires source authority.
    #[must_use]
    pub fn numerical_evidence(&self) -> &JointExperimentEvidence {
        &self.evidence
    }
}

/// Bind an executed compiler-owned adaptive full-history policy result to decision evidence.
///
/// # Errors
/// Refuses another source method, malformed coordinates, scope or covariance.
pub fn evidence_from_compiled_adaptive_regime(
    source: NativeJointDesignResult,
    evidence_id: &str,
) -> Result<CompiledAdaptiveRegimeEvidence, EvidenceRefusal> {
    let JointDesignRecipe::AdaptiveRegime { recipe } = source.prepared().recipe() else {
        return Err(EvidenceRefusal::Invalid("compiled source is not an adaptive regime".into()));
    };
    let JointNativeDesignLaw::AdaptiveRegime(law) = source.law() else {
        return Err(EvidenceRefusal::Invalid(
            "compiled adaptive regime has an incompatible law".into(),
        ));
    };
    let k = recipe.outcomes.len();
    if law.means.len() != 2 * k || law.failures.len() != law.independent_units {
        return Err(EvidenceRefusal::Invalid("adaptive regime coordinates are misaligned".into()));
    }
    let horizon = u32::try_from(recipe.episodes.first().map_or(0, |e| e.states.len()))
        .map_err(|_| EvidenceRefusal::Invalid("adaptive regime horizon too large".into()))?;
    let components = recipe
        .outcomes
        .iter()
        .map(|(name, variable_id, units)| OutcomeComponent {
            name: name.clone(),
            variable_id: variable_id.clone(),
            units: units.clone(),
        })
        .collect();
    let mut evidence = JointExperimentEvidence::new(
        evidence_id,
        &recipe.population,
        (&recipe.policies[0].id, &recipe.policies[1].id),
        horizon,
        components,
        (law.means[..k].to_vec(), law.means[k..].to_vec()),
        law.covariance
            .as_ref()
            .map_or(Dependence::PointOnly, |c| Dependence::JointArms { covariance: c.clone() }),
        DesignRecord {
            assignment_design: "adaptive_regime".into(),
            treated_units: law.independent_units,
            control_units: law.independent_units,
            outcome_rows: law.independent_units,
        },
        vec![],
        Exposure::NoBetweenUnitInterference { assumption_id: recipe.isolation.clone() },
    )?;
    evidence.design_contract_id = Some(source.dependency_identity().to_hex());
    evidence.design_details = Some(DesignDetails::AdaptiveRegime {
        subject_ids: recipe.episodes.iter().map(|e| e.id.clone()).collect(),
        periods: horizon,
        policy_prespecification_assumption_id: recipe.policy_prespecification.clone(),
        positivity_assumption_id: recipe.positivity.clone(),
        exchangeability_assumption_id: recipe.exchangeability.clone(),
        consistency_assumption_id: recipe.consistency.clone(),
        followup_assumption_id: recipe.followup.clone(),
        failures: law.failures.clone(),
    });
    Ok(CompiledAdaptiveRegimeEvidence { source, evidence })
}

/// Compile and execute a finite-history adaptive policy comparison under the caller's context.
///
/// # Errors
/// Refuses invalid assumptions or policy support, incompatible inference, cancellation or resources.
pub fn compile_and_evidence_from_adaptive_regime(
    recipe: antecedent_estimate::joint_design::adaptive_regime::JointAdaptiveRegimeRecipe,
    evidence_id: &str,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<CompiledAdaptiveRegimeEvidence, EvidenceRefusal> {
    let prepared =
        PreparedNativeJointDesign::compile(JointDesignRecipe::AdaptiveRegime { recipe }, ctx)
            .map_err(super::compiled_decision_refusal)?;
    let source = prepared
        .execute(JointDesignInference::Frequentist, ctx)
        .map_err(super::compiled_decision_refusal)?;
    evidence_from_compiled_adaptive_regime(source, evidence_id)
}
