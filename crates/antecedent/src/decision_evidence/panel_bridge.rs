//! Explicit model-conditional use of a retained randomized excursion in a panel response.
// SPDX-License-Identifier: MIT OR Apache-2.0
use super::{
    CompiledAdaptiveRegimeEvidence, CompiledClusterEvidence, CompiledExcursionEvidence,
    CompiledExperimentalEvidence, CompiledRegimeEvidence, EvidenceRefusal,
};
use crate::analysis::joint_panel::{
    JointPanelError, JointPanelResponseFunctional, JointPanelResult, PanelExperimentDecisionSource,
    PanelExperimentEvidence, PanelExperimentMapping, PanelExperimentModularPolicy,
    PanelExperimentResult, PanelExperimentUpdate, PanelJointExperimentComponent,
    PanelJointExperimentResult, PanelJointExperimentUpdate,
};
use antecedent_contract::{
    QuantityRole, ScientificQuantity,
    bridge::{AssumptionKind, BridgeContract},
    evidence_ledger::{EvidenceLedger, SourceKind, SourceStatus},
};
use antecedent_core::{CanonicalWriter, ExecutionContext, SemanticDigest};
use antecedent_design::decision_evidence::{Dependence, Exposure, JointExperimentEvidence};
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

/// The exact source and target scopes and the declared sampling-model assumption of one bridge.
fn check_excursion_translation(
    dependency: SemanticDigest,
    quantity: &ScientificQuantity,
    bridge: &BridgeContract,
    likelihood_assumption_id: &str,
) -> Result<(), JointPanelError> {
    if bridge.source.quantity != *quantity {
        return Err(JointPanelError::Compile(
            "bridge source does not match the retained excursion estimand".into(),
        ));
    }
    let translation = bridge.estimand_bridge.as_ref().ok_or_else(|| {
        JointPanelError::Compile(
            "the excursion-to-panel bridge needs an explicit estimand translation".into(),
        )
    })?;
    if translation.experiment_id != dependency.to_hex()
        || translation.experiment.population_id != quantity.population_id
        || translation.experiment.exposure_regime != quantity.regime_id
        || translation.target.population_id != bridge.target.quantity.population_id
        || translation.target.exposure_regime != bridge.target.quantity.regime_id
    {
        return Err(JointPanelError::Compile(
            "estimand translation does not bind the exact source and target scopes".into(),
        ));
    }
    if likelihood_assumption_id.trim().is_empty()
        || !bridge.assumptions.iter().any(|a| {
            a.id == likelihood_assumption_id
                && a.kind == AssumptionKind::Additional
                && !a.statement.trim().is_empty()
        })
    {
        return Err(JointPanelError::Compile(
            "the Gaussian sampling model needs its own explicit bridge assumption".into(),
        ));
    }
    Ok(())
}

fn check_source(
    source: &CompiledExcursionEvidence,
    id: antecedent_contract::evidence_ledger::SourceId,
    ledger: &EvidenceLedger,
) -> Result<(), JointPanelError> {
    check_dependency_source(source.source().dependency_identity(), id, ledger)
}

fn check_dependency_source(
    dependency: SemanticDigest,
    id: antecedent_contract::evidence_ledger::SourceId,
    ledger: &EvidenceLedger,
) -> Result<(), JointPanelError> {
    let entry = ledger.source(id).ok_or_else(|| {
        JointPanelError::Unsupported("excursion source is absent from the current ledger".into())
    })?;
    if entry.status != SourceStatus::Active
        || entry.declaration.kind != SourceKind::Experiment
        || entry.declaration.label != dependency.to_hex()
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
    check_excursion_translation(
        excursion.source().dependency_identity(),
        &quantity,
        &update.bridge,
        likelihood_assumption_id,
    )?;
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

/// A retained, executed experiment whose joint sampling evidence can inform a panel response.
///
/// Each variant keeps its compiler result, so the evidence stays attached to the exact design,
/// assignment law and dependency identity that produced it.
#[derive(Clone, Debug)]
pub enum RetainedExperiment {
    /// Sequentially randomized excursion.
    Excursion(CompiledExcursionEvidence),
    /// Matched-pair or switchback randomized experiment.
    Experiment(CompiledExperimentalEvidence),
    /// Cluster-randomized experiment.
    Cluster(CompiledClusterEvidence),
    /// Static treatment regime.
    StaticRegime(CompiledRegimeEvidence),
    /// History-adaptive policy comparison.
    AdaptiveRegime(CompiledAdaptiveRegimeEvidence),
}
macro_rules! retained_from {
    ($($variant:ident($evidence:ty)),*) => {$(
        impl From<$evidence> for RetainedExperiment {
            fn from(evidence: $evidence) -> Self {
                Self::$variant(evidence)
            }
        }
    )*};
}
retained_from!(
    Excursion(CompiledExcursionEvidence),
    Experiment(CompiledExperimentalEvidence),
    Cluster(CompiledClusterEvidence),
    StaticRegime(CompiledRegimeEvidence),
    AdaptiveRegime(CompiledAdaptiveRegimeEvidence)
);
impl RetainedExperiment {
    /// The decision-evidence record the compiler produced.
    #[must_use]
    pub fn evidence(&self) -> &JointExperimentEvidence {
        match self {
            Self::Excursion(e) => e.numerical_evidence(),
            Self::Experiment(e) => e.numerical_evidence(),
            Self::Cluster(e) => e.numerical_evidence(),
            Self::StaticRegime(e) => e.numerical_evidence(),
            Self::AdaptiveRegime(e) => e.numerical_evidence(),
        }
    }
    /// Dependency identity of the retained design and its execution.
    #[must_use]
    pub fn dependency_identity(&self) -> SemanticDigest {
        match self {
            Self::Excursion(e) => e.source().dependency_identity(),
            Self::Experiment(e) => e.source().dependency_identity(),
            Self::Cluster(e) => e.source().dependency_identity(),
            Self::StaticRegime(e) => e.source().dependency_identity(),
            Self::AdaptiveRegime(e) => e.source().dependency_identity(),
        }
    }
    /// The exact experimental quantity of one component, for an explicit bridge.
    ///
    /// # Errors
    /// The component is absent, or the evidence carries no design identity to bind it to.
    pub fn component_quantity(
        &self,
        component: &str,
    ) -> Result<ScientificQuantity, EvidenceRefusal> {
        let functional = match self {
            Self::Excursion(e) => return compiled_excursion_quantity(e, component),
            Self::Experiment(_) => "randomized_experiment_mean_contrast",
            Self::Cluster(_) => "cluster_randomized_mean_contrast",
            Self::StaticRegime(_) => "static_regime_mean_contrast",
            Self::AdaptiveRegime(_) => "adaptive_regime_mean_contrast",
        };
        let evidence = self.evidence();
        let outcome = evidence
            .components
            .iter()
            .find(|c| c.name == component)
            .ok_or_else(|| EvidenceRefusal::Invalid("evidence component is absent".into()))?;
        let design = evidence.design_contract_id.as_deref().ok_or_else(|| {
            EvidenceRefusal::Invalid("retained evidence carries no design identity".into())
        })?;
        let mut w = CanonicalWriter::new("antecedent.decision_evidence.retained_target.v1");
        w.text(&evidence.treated_regime_id)
            .text(&evidence.control_regime_id)
            .text(&evidence.design.assignment_design)
            .text(design);
        Ok(ScientificQuantity {
            variable_id: outcome.variable_id.clone(),
            variable_name: outcome.name.clone(),
            role: QuantityRole::Outcome,
            units: outcome.units.clone(),
            population_id: evidence.population_id.clone(),
            regime_id: format!("{functional}:{}", w.finish_hex()),
            horizon: evidence.horizon,
            functional_id: functional.into(),
            conditioning: vec![],
            transform_id: "identity".into(),
        })
    }
}

/// One evidence component and its licensed translation to a retained response functional.
#[derive(Clone, Debug)]
pub struct JointEvidenceComponent {
    /// Name of the evidence outcome component.
    pub component: String,
    /// Licensed translation; its source must be [`RetainedExperiment::component_quantity`].
    pub bridge: BridgeContract,
    /// Time functional of the retained worlds that the bridge targets.
    pub functional: JointPanelResponseFunctional,
    /// One affine realization per retained draw slot, paired with the draws.
    pub mapping: PanelExperimentMapping,
}

/// Observed contrasts and their joint sampling law, as the bridge can use them.
#[derive(Clone, Debug, PartialEq)]
pub struct JointContrastSampling {
    /// Observed treated-minus-control contrast of each named component.
    pub observed: Vec<f64>,
    /// Joint covariance of those contrast estimators, row-major.
    pub covariance: Vec<f64>,
    /// Assumed absolute bias of each contrast when the evidence is only assumption-bounded.
    pub contrast_bias_bounds: Option<Vec<f64>>,
}

/// Observed contrasts and their joint sampling covariance for the named components.
///
/// The covariance is the retained evidence's own joint law of the arm-mean estimators, mapped to
/// the treated-minus-control contrasts. Evidence that carries no joint covariance cannot produce
/// one: marginal variances do not determine dependence, and point estimates carry no sampling law.
/// Evidence whose exposure is only bounded is returned with its assumed bias bounds; those are an
/// assumption range and never become a likelihood.
///
/// # Errors
/// An absent or repeated component, evidence whose dependence is not a joint covariance, or an
/// unspecified exposure.
pub fn joint_contrast_sampling(
    evidence: &JointExperimentEvidence,
    components: &[&str],
) -> Result<JointContrastSampling, JointPanelError> {
    let k = evidence.components.len();
    let mut index = Vec::with_capacity(components.len());
    for name in components {
        let i = evidence
            .components
            .iter()
            .position(|c| c.name == *name)
            .ok_or_else(|| JointPanelError::Compile("evidence component is absent".into()))?;
        if index.contains(&i) {
            return Err(JointPanelError::Unsupported(
                "a component is named twice; counting it twice would overstate the evidence".into(),
            ));
        }
        index.push(i);
    }
    // Cov(contrast_i, contrast_j) from the arm-mean covariance, entry by entry.
    let contrast_covariance = |i: usize, j: usize| {
        match &evidence.dependence {
        Dependence::JointArms { covariance } => {
            let w = 2 * k;
            Ok(covariance[i * w + j] + covariance[(k + i) * w + k + j]
                - covariance[i * w + k + j]
                - covariance[(k + i) * w + j])
        }
        Dependence::Joint { treated, control } => Ok(treated[i * k + j] + control[i * k + j]),
        Dependence::MarginalsOnly { .. } => Err(JointPanelError::Unsupported(
            "the evidence has marginal variances only, so the dependence between components is unknown and stays unknown. Remedy: supply evidence with a joint arm covariance, or update one component at a time with `condition_panel_on_compiled_excursion`".into(),
        )),
        Dependence::PointOnly => Err(JointPanelError::Unsupported(
            "the evidence is point-only and carries no sampling law. Remedy: supply evidence with a joint arm covariance (for example a bootstrap or analytic covariance)".into(),
        )),
    }
    };
    let contrast_bias_bounds = match &evidence.exposure {
        Exposure::NoBetweenUnitInterference { .. } => None,
        Exposure::BoundedSpillover { bias_bounds, .. } => {
            Some(index.iter().map(|&i| bias_bounds[i]).collect())
        }
        Exposure::BoundedArmSpillover { treated_bias_bounds, control_bias_bounds, .. } => {
            Some(index.iter().map(|&i| treated_bias_bounds[i] + control_bias_bounds[i]).collect())
        }
        Exposure::Unspecified => {
            return Err(JointPanelError::Unsupported(
                "the evidence declares no exposure model. Remedy: declare between-unit interference or a bound on it".into(),
            ));
        }
    };
    let observed =
        index.iter().map(|&i| evidence.treated_means[i] - evidence.control_means[i]).collect();
    let mut covariance = Vec::with_capacity(index.len() * index.len());
    for &i in &index {
        for &j in &index {
            covariance.push(contrast_covariance(i, j)?);
        }
    }
    Ok(JointContrastSampling { observed, covariance, contrast_bias_bounds })
}

/// Joint conditional panel update retaining its exact retained experiment and likelihood declaration.
#[derive(Clone, Debug)]
pub struct RetainedJointPanelUpdate {
    source: RetainedExperiment,
    result: PanelJointExperimentResult,
    likelihood_assumption_id: String,
    experiment_id: antecedent_contract::evidence_ledger::SourceId,
}
impl RetainedJointPanelUpdate {
    /// Source remains attached to the model update.
    #[must_use]
    pub fn experiment(&self) -> &RetainedExperiment {
        &self.source
    }
    /// The existing joint panel owner; consuming it needs the experiment's current authority too.
    #[must_use]
    pub fn result(&self) -> &PanelJointExperimentResult {
        &self.result
    }
    /// Explicit assumption that supplies the Gaussian sampling model.
    #[must_use]
    pub fn likelihood_assumption_id(&self) -> &str {
        &self.likelihood_assumption_id
    }
    /// The updated law of one component after checking current source and ledger authority.
    ///
    /// # Errors
    /// Withdrawal, changed source identity, invalid bridge or other current-authority refusal.
    pub fn law_source(
        &self,
        component: usize,
        ledger: &EvidenceLedger,
    ) -> Result<PanelExperimentDecisionSource, JointPanelError> {
        check_dependency_source(self.source.dependency_identity(), self.experiment_id, ledger)?;
        self.result.law_source(component, ledger)
    }
}

/// Condition a panel posterior jointly on several components of one retained experiment.
///
/// Observed contrasts and their covariance come from the retained evidence, never from the
/// caller, so the update cannot disagree with the experiment or silently assume independence.
/// Each component's bridge passes the same exact source, target, likelihood-assumption and
/// ledger checks as [`condition_panel_on_compiled_excursion`]. The Gaussian sampling model is a
/// separate declared assumption; the result stays model-conditional and uncalibrated, and only
/// the named targets are informed. Evidence whose interference is only bounded yields an
/// assumption-bounded result that answers sensitivity questions, not decisions.
///
/// # Errors
/// Evidence without a joint covariance, a mismatched source, target or likelihood declaration,
/// or any refusal of [`PanelJointExperimentResult`].
pub fn condition_panel_jointly_on_retained_evidence(
    panel: &JointPanelResult,
    experiment: impl Into<RetainedExperiment>,
    components: Vec<JointEvidenceComponent>,
    evidence: PanelExperimentEvidence,
    likelihood_assumption_id: &str,
    ledger: &EvidenceLedger,
    ctx: &ExecutionContext,
) -> Result<RetainedJointPanelUpdate, JointPanelError> {
    let experiment = experiment.into();
    let names: Vec<&str> = components.iter().map(|c| c.component.as_str()).collect();
    let sampling = joint_contrast_sampling(experiment.evidence(), &names)?;
    for c in &components {
        let quantity = experiment
            .component_quantity(&c.component)
            .map_err(|e| JointPanelError::Compile(e.to_string()))?;
        check_excursion_translation(
            experiment.dependency_identity(),
            &quantity,
            &c.bridge,
            likelihood_assumption_id,
        )?;
    }
    check_dependency_source(experiment.dependency_identity(), evidence.experiment, ledger)?;
    let design = &experiment.evidence().design;
    let sample_size = u64::try_from(design.treated_units.min(design.control_units))
        .map_err(|_| JointPanelError::Compile("independent-unit count overflow".into()))?;
    let experiment_id = evidence.experiment;
    let result = panel.condition_joint_experiment(
        PanelJointExperimentUpdate {
            components: components
                .into_iter()
                .map(|c| PanelJointExperimentComponent {
                    bridge: c.bridge,
                    functional: c.functional,
                    mapping: c.mapping,
                })
                .collect(),
            observed_means: sampling.observed,
            sampling_covariance: sampling.covariance,
            sample_size,
            modular_policy: PanelExperimentModularPolicy::FrozenJointImportance,
            evidence,
            contrast_bias_bounds: sampling.contrast_bias_bounds,
        },
        ledger,
        ctx,
    )?;
    Ok(RetainedJointPanelUpdate {
        source: experiment,
        result,
        likelihood_assumption_id: likelihood_assumption_id.into(),
        experiment_id,
    })
}

#[cfg(test)]
mod tests {
    use super::{Dependence, JointExperimentEvidence, joint_contrast_sampling};
    use antecedent_design::decision_evidence::{DesignRecord, Exposure, OutcomeComponent};

    fn evidence(dependence: Dependence) -> JointExperimentEvidence {
        let component = |name: &str| OutcomeComponent {
            name: name.into(),
            variable_id: format!("v-{name}"),
            units: "unit".into(),
        };
        JointExperimentEvidence::new(
            "e",
            "population",
            ("on", "off"),
            1,
            vec![component("a"), component("b")],
            (vec![3.0, 5.0], vec![1.0, 4.0]),
            dependence,
            DesignRecord {
                assignment_design: "complete".into(),
                treated_units: 10,
                control_units: 10,
                outcome_rows: 20,
            },
            vec![],
            Exposure::NoBetweenUnitInterference { assumption_id: "isolation".into() },
        )
        .unwrap()
    }

    #[test]
    fn contrast_covariance_is_the_arm_covariance_mapped_to_treated_minus_control() {
        // Arm order: treated a, treated b, control a, control b.
        #[rustfmt::skip]
        let arms = vec![
            4.0, 1.0, 0.5, 0.0,
            1.0, 3.0, 0.0, 0.25,
            0.5, 0.0, 2.0, 0.5,
            0.0, 0.25, 0.5, 1.0,
        ];
        let sampling = joint_contrast_sampling(
            &evidence(Dependence::JointArms { covariance: arms }),
            &["b", "a"],
        )
        .unwrap();
        let (observed, covariance) = (sampling.observed, sampling.covariance);
        assert_eq!(sampling.contrast_bias_bounds, None);
        assert_eq!(observed, vec![1.0, 2.0]);
        // Var(b) = 3 + 1 - 2 * 0.25, Var(a) = 4 + 2 - 2 * 0.5, Cov(b, a) = 1 + 0.5 - 0 - 0.
        assert_eq!(covariance, vec![3.5, 1.5, 1.5, 5.0]);
        let independent_arms = joint_contrast_sampling(
            &evidence(Dependence::Joint {
                treated: vec![4.0, 1.0, 1.0, 3.0],
                control: vec![2.0, 0.5, 0.5, 1.0],
            }),
            &["a", "b"],
        )
        .unwrap()
        .covariance;
        assert_eq!(independent_arms, vec![6.0, 1.5, 1.5, 4.0]);
    }

    #[test]
    fn bounded_interference_stays_an_assumption_range_and_never_a_likelihood() {
        let mut bounded = evidence(Dependence::Joint {
            treated: vec![1.0, 0.0, 0.0, 1.0],
            control: vec![1.0, 0.0, 0.0, 1.0],
        });
        bounded.exposure = Exposure::BoundedSpillover {
            assumption_id: "spill".into(),
            bias_bounds: vec![0.1, 0.2],
        };
        let sampling = joint_contrast_sampling(&bounded, &["b", "a"]).unwrap();
        assert_eq!(sampling.contrast_bias_bounds, Some(vec![0.2, 0.1]));
        bounded.exposure = Exposure::BoundedArmSpillover {
            assumption_id: "spill".into(),
            treated_bias_bounds: vec![0.1, 0.2],
            control_bias_bounds: vec![0.3, 0.4],
        };
        let sampling = joint_contrast_sampling(&bounded, &["a", "b"]).unwrap();
        let bounds = sampling.contrast_bias_bounds.unwrap();
        assert!((bounds[0] - 0.4).abs() < 1e-12 && (bounds[1] - 0.6).abs() < 1e-12);
        bounded.exposure = Exposure::Unspecified;
        assert!(joint_contrast_sampling(&bounded, &["a", "b"]).is_err());
    }

    #[test]
    fn evidence_without_a_joint_covariance_never_becomes_one() {
        let marginals =
            Dependence::MarginalsOnly { treated: vec![1.0, 1.0], control: vec![1.0, 1.0] };
        for dependence in [marginals, Dependence::PointOnly] {
            let refused = joint_contrast_sampling(&evidence(dependence), &["a", "b"]).unwrap_err();
            assert!(format!("{refused}").contains("Remedy"), "{refused}");
        }
        let arms = Dependence::Joint {
            treated: vec![1.0, 0.0, 0.0, 1.0],
            control: vec![1.0, 0.0, 0.0, 1.0],
        };
        assert!(joint_contrast_sampling(&evidence(arms.clone()), &["a", "a"]).is_err());
        assert!(joint_contrast_sampling(&evidence(arms), &["a", "missing"]).is_err());
    }
}
