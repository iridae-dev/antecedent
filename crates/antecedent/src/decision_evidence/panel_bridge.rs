//! Explicit model-conditional use of a retained randomized excursion in a panel response.
// SPDX-License-Identifier: MIT OR Apache-2.0
use super::{CompiledExcursionEvidence, EvidenceRefusal};
use crate::analysis::joint_panel::{
    JointPanelError, JointPanelResult, PanelExperimentDecisionSource, PanelExperimentResult,
    PanelExperimentUpdate,
};
use antecedent_contract::{
    QuantityRole, ScientificQuantity,
    bridge::AssumptionKind,
    evidence_ledger::{EvidenceLedger, SourceKind, SourceStatus},
};
use antecedent_core::{CanonicalWriter, ExecutionContext};
use antecedent_estimate::joint_design::compiled::JointDesignRecipe;

/// The exact source quantity of one retained excursion component, for an explicit bridge.
///
/// # Errors
/// The named outcome is absent or the retained source is not an excursion.
pub fn compiled_excursion_quantity(
    evidence: &CompiledExcursionEvidence,
    component: &str,
) -> Result<ScientificQuantity, EvidenceRefusal> {
    let JointDesignRecipe::Excursion { recipe } = evidence.source().prepared().recipe() else {
        return Err(EvidenceRefusal::Invalid("source is not an excursion".into()));
    };
    let outcome = recipe
        .outcomes
        .iter()
        .find(|o| o.component_name == component)
        .ok_or_else(|| EvidenceRefusal::Invalid("excursion component is absent".into()))?;
    let mut w = CanonicalWriter::new("antecedent.decision_evidence.excursion_target.v1");
    w.text(&recipe.treated_regime_id)
        .text(&recipe.control_regime_id)
        .word(u64::from(recipe.lag))
        .text(&recipe.assignment_law);
    Ok(ScientificQuantity {
        variable_id: outcome.variable_id.clone(),
        variable_name: outcome.component_name.clone(),
        role: QuantityRole::Outcome,
        units: outcome.units.clone(),
        population_id: recipe.population.clone(),
        regime_id: format!("excursion:{}", w.finish_hex()),
        horizon: recipe.horizon,
        functional_id: "sequential_randomized_excursion_mean_contrast".into(),
        conditioning: vec![],
        transform_id: "identity".into(),
    })
}

/// Conditional panel update retaining its exact compiled excursion and likelihood declaration.
#[derive(Clone, Debug)]
pub struct ExcursionPanelUpdate {
    source: CompiledExcursionEvidence,
    result: PanelExperimentResult,
    likelihood_assumption_id: String,
    experiment_id: antecedent_contract::evidence_ledger::SourceId,
}
impl ExcursionPanelUpdate {
    /// Source remains attached to the model update.
    #[must_use]
    pub fn excursion(&self) -> &CompiledExcursionEvidence {
        &self.source
    }
    /// Split into the existing panel owner and retained excursion owner; both must stay live.
    #[must_use]
    pub fn into_parts(self) -> (PanelExperimentResult, CompiledExcursionEvidence) {
        (self.result, self.source)
    }
    /// Explicit assumption that supplies the Gaussian sampling model.
    #[must_use]
    pub fn likelihood_assumption_id(&self) -> &str {
        &self.likelihood_assumption_id
    }
    /// Obtain the existing model-conditional law after checking current source and ledger authority.
    ///
    /// # Errors
    /// Withdrawal, changed source identity, invalid bridge or other current-authority refusal.
    pub fn law_source(
        &self,
        ledger: &EvidenceLedger,
    ) -> Result<PanelExperimentDecisionSource, JointPanelError> {
        check_source(&self.source, self.experiment_id, ledger)?;
        self.result.law_source(ledger)
    }
}

fn check_source(
    source: &CompiledExcursionEvidence,
    id: antecedent_contract::evidence_ledger::SourceId,
    ledger: &EvidenceLedger,
) -> Result<(), JointPanelError> {
    let entry = ledger.source(id).ok_or_else(|| {
        JointPanelError::Unsupported("excursion source is absent from the current ledger".into())
    })?;
    if entry.status != SourceStatus::Active
        || entry.declaration.kind != SourceKind::Experiment
        || entry.declaration.label != source.source().dependency_identity().to_hex()
    {
        return Err(JointPanelError::Unsupported(
            "current experiment source does not match the retained excursion or is withdrawn"
                .into(),
        ));
    }
    Ok(())
}

/// Condition an existing panel posterior on a retained excursion using its existing experiment bridge.
///
/// The bridge must name [`compiled_excursion_quantity`] exactly and an explicit `Additional`
/// assumption for the Gaussian sampling model. The caller declares its known per-unit variance;
/// this function never substitutes the excursion's estimated covariance as a known likelihood.
/// The observed mean and independent-unit count must equal the retained excursion. The existing
/// bridge, overlap, mapping, posterior and model-conditional authority checks remain in force.
/// This does not establish that an excursion equals a sustained or adaptive policy effect.
///
/// # Errors
/// Mismatched source, target, likelihood declaration, evidence, data, bridge, or execution policy.
#[allow(
    clippy::float_cmp,
    reason = "exact equality preserves the declared boundary and identity semantics"
)]
pub fn condition_panel_on_compiled_excursion(
    panel: &JointPanelResult,
    excursion: CompiledExcursionEvidence,
    component: &str,
    update: PanelExperimentUpdate,
    likelihood_assumption_id: &str,
    ledger: &EvidenceLedger,
    ctx: &ExecutionContext,
) -> Result<ExcursionPanelUpdate, JointPanelError> {
    let quantity = compiled_excursion_quantity(&excursion, component)
        .map_err(|e| JointPanelError::Compile(e.to_string()))?;
    if update.bridge.source.quantity != quantity {
        return Err(JointPanelError::Compile(
            "bridge source does not match the retained excursion estimand".into(),
        ));
    }
    let translation = update.bridge.estimand_bridge.as_ref().ok_or_else(|| {
        JointPanelError::Compile(
            "the excursion-to-panel bridge needs an explicit estimand translation".into(),
        )
    })?;
    if translation.experiment_id != excursion.source().dependency_identity().to_hex()
        || translation.experiment.population_id != quantity.population_id
        || translation.experiment.exposure_regime != quantity.regime_id
        || translation.target.population_id != update.bridge.target.quantity.population_id
        || translation.target.exposure_regime != update.bridge.target.quantity.regime_id
    {
        return Err(JointPanelError::Compile(
            "estimand translation does not bind the exact source and target scopes".into(),
        ));
    }
    if likelihood_assumption_id.trim().is_empty()
        || !update.bridge.assumptions.iter().any(|a| {
            a.id == likelihood_assumption_id
                && a.kind == AssumptionKind::Additional
                && !a.statement.trim().is_empty()
        })
    {
        return Err(JointPanelError::Compile(
            "the Gaussian sampling model needs its own explicit bridge assumption".into(),
        ));
    }
    let numerical = excursion.numerical_evidence();
    let index = numerical
        .components
        .iter()
        .position(|c| c.name == component)
        .ok_or_else(|| JointPanelError::Compile("excursion component is absent".into()))?;
    let observed = numerical.treated_means[index] - numerical.control_means[index];
    if update.observed_mean != observed
        || usize::try_from(update.sample_size).ok() != Some(numerical.design.treated_units)
    {
        return Err(JointPanelError::Compile(
            "experiment mean or independent-unit count differs from the retained excursion".into(),
        ));
    }
    check_source(&excursion, update.evidence.experiment, ledger)?;
    let experiment_id = update.evidence.experiment;
    let result = panel.condition_experiment(update, ledger, ctx)?;
    Ok(ExcursionPanelUpdate {
        source: excursion,
        result,
        likelihood_assumption_id: likelihood_assumption_id.into(),
        experiment_id,
    })
}
