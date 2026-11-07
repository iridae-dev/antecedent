//! Experimental evidence for decision-relevant outcomes.
//!
//! The estimator side ([`antecedent_estimate::complete_cluster_joint_contrast`]) and the
//! decision side ([`antecedent_design::decision_evidence`]) are separate layers; this module is
//! the one place they meet, so the design crate stays independent of the estimators.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

mod adaptive;
mod designs;

pub use designs::{
    CompiledJointExcursion, CompiledJointStaticRegime, CompiledJointSwitchback,
    CompiledMatchedPairDesign, JointExcursionDesign, JointStaticRegimeDesign,
    JointSwitchbackDesign, MatchedPairDesign, compile_joint_excursion, compile_joint_static_regime,
    compile_joint_switchback, compile_matched_pair_design, evidence_from_joint_excursion,
    evidence_from_joint_static_regime, evidence_from_joint_switchback, evidence_from_matched_pairs,
};
mod panel_bridge;
pub use adaptive::{
    CompiledAdaptiveRegimeEvidence, compile_and_evidence_from_adaptive_regime,
    evidence_from_compiled_adaptive_regime,
};
pub use antecedent_design::decision_evidence::*;
pub use antecedent_estimate::randomized_joint::{
    JointClusterContrast, centered_fixed_cuped_totals, complete_cluster_joint_contrast,
};
pub use panel_bridge::{
    ExcursionPanelUpdate, JointContrastSampling, JointEvidenceComponent, RetainedExperiment,
    RetainedJointPanelUpdate, compiled_excursion_quantity,
    condition_panel_jointly_on_retained_evidence, condition_panel_on_compiled_excursion,
    joint_contrast_sampling,
};

/// Wrap a historical joint cluster estimate as numerical decision evidence.
///
/// This compatibility function cannot reconstruct the original source or compiler proof and
/// grants no current scientific authority. Use [`compile_and_evidence_from_cluster`] for an
/// owned, source-bound result that retains admission and supports current-source checks.
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

/// Joint excursion evidence retaining its native compiler result and resource ownership.
///
/// The numerical evidence alone is a historical component; causal consumption must keep this
/// owner and check current evidence authority. No support or calibration support is implied.
#[derive(Clone, Debug)]
pub struct CompiledExcursionEvidence {
    source: crate::analysis::joint_panel::designs::NativeJointDesignResult,
    evidence: JointExperimentEvidence,
}

impl CompiledExcursionEvidence {
    /// The retained source, including its method, premises and dependency identity.
    #[must_use]
    pub fn source(&self) -> &crate::analysis::joint_panel::designs::NativeJointDesignResult {
        &self.source
    }

    /// Historical numerical evidence. A caller must separately check current ledger authority.
    #[must_use]
    pub fn numerical_evidence(&self) -> &JointExperimentEvidence {
        &self.evidence
    }
}

/// Bind an executed, retained compiler excursion law to the existing decision evidence contract.
///
/// Outcome IDs and units come from the source recipe and cannot be replaced by labels supplied
/// by the caller. This does not turn the excursion into a whole-program intervention.
///
/// # Errors
/// Refuses other compiler recipes, incompatible laws, malformed evidence labels or unsupported
/// decision-evidence scope.
pub fn evidence_from_compiled_excursion(
    source: crate::analysis::joint_panel::designs::NativeJointDesignResult,
    evidence_id: &str,
    regimes: (&str, &str),
    horizon: u32,
) -> Result<CompiledExcursionEvidence, EvidenceRefusal> {
    use antecedent_estimate::joint_design::compiled::{JointDesignRecipe, JointNativeDesignLaw};
    let JointDesignRecipe::Excursion { recipe } = source.prepared().recipe() else {
        return Err(EvidenceRefusal::Invalid("compiled source is not an excursion".into()));
    };
    let JointNativeDesignLaw::Excursion(law) = source.law() else {
        return Err(EvidenceRefusal::Invalid("compiled excursion has an incompatible law".into()));
    };
    if regimes != (recipe.treated_regime_id.as_str(), recipe.control_regime_id.as_str()) {
        return Err(EvidenceRefusal::Invalid(
            "requested excursion regimes differ from the compiler source".into(),
        ));
    }
    if horizon != recipe.horizon {
        return Err(EvidenceRefusal::Invalid(
            "requested horizon differs from the compiler source".into(),
        ));
    }
    let k = recipe.outcomes.len();
    let width = k
        .checked_mul(2)
        .ok_or_else(|| EvidenceRefusal::Invalid("excursion coordinate overflow".into()))?;
    let covariance_size = width
        .checked_mul(width)
        .ok_or_else(|| EvidenceRefusal::Invalid("excursion covariance overflow".into()))?;
    if law.means.len() != width || law.covariance.len() != covariance_size {
        return Err(EvidenceRefusal::Invalid(
            "compiled excursion coordinates are misaligned".into(),
        ));
    }
    let components = recipe
        .outcomes
        .iter()
        .map(|outcome| OutcomeComponent {
            name: outcome.component_name.clone(),
            variable_id: outcome.variable_id.clone(),
            units: outcome.units.clone(),
        })
        .collect();
    let mut evidence = JointExperimentEvidence::new(
        evidence_id,
        &recipe.population,
        regimes,
        horizon,
        components,
        (law.means[..k].to_vec(), law.means[k..].to_vec()),
        Dependence::JointArms { covariance: law.covariance.clone() },
        DesignRecord {
            assignment_design: "temporal_excursion".into(),
            treated_units: law.independent_units,
            control_units: law.independent_units,
            outcome_rows: recipe.unit_ids.len(),
        },
        vec![],
        Exposure::NoBetweenUnitInterference { assumption_id: recipe.isolation.clone() },
    )?;
    evidence.design_contract_id = Some(source.dependency_identity().to_hex());
    evidence.design_details = Some(DesignDetails::TemporalExcursion {
        unit_ids: recipe.unit_ids.clone(),
        periods: recipe.periods.clone(),
        treatment_probabilities: recipe.treatment_probabilities.clone(),
        realized_assignment: recipe.assignments.clone(),
        lag: recipe.lag,
        min_probability: recipe.min_probability,
        assignment_law_id: recipe.assignment_law.clone(),
        independent_units_assumption_id: recipe.independence.clone(),
        administrative_followup_assumption_id: recipe.followup.clone(),
        consistency_assumption_id: recipe.consistency.clone(),
        paired_decisions: law.paired_decisions,
    });
    Ok(CompiledExcursionEvidence { source, evidence })
}

/// Compile and execute a joint excursion while retaining the compiler result and its admission.
///
/// # Errors
/// Refuses changed designs, incompatible outcomes, absent assumptions, unsupported assignment,
/// cancellation or resource admission failure.
#[allow(
    clippy::too_many_arguments,
    clippy::needless_pass_by_value,
    reason = "the public API accepts owned outcome declarations"
)]
pub fn compile_and_evidence_from_joint_excursion(
    design: &CompiledJointExcursion,
    outcomes: &[&[f64]],
    assigned: &[bool],
    evidence_id: &str,
    population_id: &str,
    regimes: (&str, &str),
    horizon: u32,
    components: Vec<OutcomeComponent>,
    exposure: Exposure,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<CompiledExcursionEvidence, EvidenceRefusal> {
    use crate::analysis::joint_panel::designs::PreparedNativeJointDesign;
    use antecedent_estimate::joint_design::{
        compiled::{JointDesignInference, JointDesignRecipe},
        excursion::{ExcursionOutcome, JointExcursionRecipe},
    };
    let n = design.design.unit_ids.len();
    if assigned.len() != n
        || outcomes.is_empty()
        || components.len() != outcomes.len()
        || outcomes.iter().any(|v| v.len() != n)
    {
        return Err(EvidenceRefusal::Invalid(
            "excursion inputs must align before source preparation".into(),
        ));
    }
    let _source_admission = admit_temporal_source(ctx, &design.design.unit_ids, outcomes.len(), 0)?;
    let checked = compile_joint_excursion(design.design.clone())?;
    if checked.identity != design.identity
        || checked.paired_decisions != design.paired_decisions
        || checked.independent_units != design.independent_units
    {
        return Err(EvidenceRefusal::Invalid("compiled excursion design was modified".into()));
    }
    let Exposure::NoBetweenUnitInterference { assumption_id } = exposure else {
        return Err(EvidenceRefusal::Invalid(
            "excursion requires no between-unit interference".into(),
        ));
    };
    if components.len() != outcomes.len() {
        return Err(EvidenceRefusal::Invalid(
            "excursion components and outcomes must align".into(),
        ));
    }
    let recipe = JointExcursionRecipe {
        population: population_id.into(),
        treated_regime_id: regimes.0.into(),
        control_regime_id: regimes.1.into(),
        unit_ids: design.design.unit_ids.clone(),
        periods: design.design.periods.clone(),
        treatment_probabilities: design.design.treatment_probabilities.clone(),
        assignments: assigned.to_vec(),
        outcomes: components
            .iter()
            .zip(outcomes)
            .map(|(c, values)| ExcursionOutcome {
                component_name: c.name.clone(),
                variable_id: c.variable_id.clone(),
                units: c.units.clone(),
                values: values.to_vec(),
            })
            .collect(),
        lag: design.design.lag,
        horizon,
        min_probability: design.design.min_probability,
        assignment_law: design.design.assignment_law_id.clone(),
        independence: design.design.independent_units_assumption_id.clone(),
        followup: design.design.administrative_followup_assumption_id.clone(),
        consistency: design.design.consistency_assumption_id.clone(),
        isolation: assumption_id,
    };
    let prepared = PreparedNativeJointDesign::compile(JointDesignRecipe::Excursion { recipe }, ctx)
        .map_err(compiled_decision_refusal)?;
    let result = prepared
        .execute(JointDesignInference::Frequentist, ctx)
        .map_err(compiled_decision_refusal)?;
    evidence_from_compiled_excursion(result, evidence_id, regimes, horizon)
}

/// Point-only regime evidence retaining its compiler result and admission.
#[derive(Clone, Debug)]
pub struct CompiledRegimeEvidence {
    source: crate::analysis::joint_panel::designs::NativeJointDesignResult,
    evidence: JointExperimentEvidence,
}
impl CompiledRegimeEvidence {
    /// Retained native source and its exact dependency.
    #[must_use]
    pub fn source(&self) -> &crate::analysis::joint_panel::designs::NativeJointDesignResult {
        &self.source
    }
    /// Historical point evidence; current authority requires an independent source check.
    #[must_use]
    pub fn numerical_evidence(&self) -> &JointExperimentEvidence {
        &self.evidence
    }
}

/// Bind a retained native static-regime result without inventing joint uncertainty.
/// # Errors
/// Refuses incompatible source, coordinate dimensions or evidence declaration.
pub fn evidence_from_compiled_static_regime(
    source: crate::analysis::joint_panel::designs::NativeJointDesignResult,
    evidence_id: &str,
    regimes: (&str, &str),
) -> Result<CompiledRegimeEvidence, EvidenceRefusal> {
    use antecedent_estimate::joint_design::compiled::{JointDesignRecipe, JointNativeDesignLaw};
    let JointDesignRecipe::StaticRegime { recipe } = source.prepared().recipe() else {
        return Err(EvidenceRefusal::Invalid("compiled source is not a static regime".into()));
    };
    let (law, covariance, bootstrap_replicates) = match source.law() {
        JointNativeDesignLaw::StaticRegime(point) => (point, None, None),
        JointNativeDesignLaw::StaticRegimeBootstrap(bootstrap) => {
            (&bootstrap.point, bootstrap.covariance.as_ref(), Some(bootstrap.realizations.len()))
        }
        _ => {
            return Err(EvidenceRefusal::Invalid(
                "compiled static regime has an incompatible law".into(),
            ));
        }
    };
    if regimes != (recipe.treated_regime_id.as_str(), recipe.control_regime_id.as_str()) {
        return Err(EvidenceRefusal::Invalid(
            "requested static regimes differ from the compiler source".into(),
        ));
    }
    let k = recipe.outcomes.len();
    if k.checked_mul(2) != Some(law.means.len()) || law.minimum_probabilities.len() != k {
        return Err(EvidenceRefusal::Invalid("compiled regime coordinates are misaligned".into()));
    }
    let horizon = u32::try_from(recipe.active_path.len())
        .map_err(|_| EvidenceRefusal::Invalid("regime horizon too large".into()))?;
    let components = recipe
        .outcomes
        .iter()
        .map(|o| OutcomeComponent {
            name: o.component_name.clone(),
            variable_id: o.variable_id.clone(),
            units: o.units.clone(),
        })
        .collect();
    let mut evidence = JointExperimentEvidence::new(
        evidence_id,
        &recipe.population,
        regimes,
        horizon,
        components,
        (law.means[..k].to_vec(), law.means[k..].to_vec()),
        covariance
            .map_or(Dependence::PointOnly, |v| Dependence::JointArms { covariance: v.clone() }),
        DesignRecord {
            assignment_design: "temporal_regime".into(),
            treated_units: law.independent_units,
            control_units: law.independent_units,
            outcome_rows: recipe.unit_ids.len(),
        },
        vec![],
        Exposure::NoBetweenUnitInterference { assumption_id: recipe.isolation.clone() },
    )?;
    evidence.design_contract_id = Some(source.dependency_identity().to_hex());
    evidence.design_details = Some(DesignDetails::TemporalRegime {
        unit_ids: recipe.unit_ids.clone(),
        periods: recipe.periods.clone(),
        realized_assignment: recipe.assignments.clone(),
        observed_minimum_probabilities: law.minimum_probabilities.clone(),
        confounder_ids: recipe.confounder_ids.clone(),
        active_path: recipe.active_path.clone(),
        reference_path: recipe.reference_path.clone(),
        method: recipe.method.clone(),
        min_probability: recipe.min_probability,
        positivity_assumption_id: recipe.positivity.clone(),
        exchangeability_assumption_id: recipe.exchangeability.clone(),
        consistency_assumption_id: recipe.consistency.clone(),
        followup_assumption_id: recipe.followup.clone(),
        model_assumption_id: recipe.model_assumption.clone(),
        bootstrap_replicates,
    });
    Ok(CompiledRegimeEvidence { source, evidence })
}

/// Compile and execute a retained multi-outcome static-regime point functional.
/// # Errors
/// Invalid source, unsupported assumptions or method, cancellation, or admission failure.
#[allow(clippy::too_many_arguments)]
pub fn compile_and_evidence_from_joint_static_regime(
    design: &CompiledJointStaticRegime,
    outcomes: &[&[f64]],
    confounders: &[Vec<f64>],
    evidence_id: &str,
    population_id: &str,
    regimes: (&str, &str),
    components: Vec<OutcomeComponent>,
    exposure: Exposure,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<CompiledRegimeEvidence, EvidenceRefusal> {
    compile_static_regime_with_inference(
        design,
        outcomes,
        confounders,
        evidence_id,
        population_id,
        regimes,
        components,
        exposure,
        ctx,
        antecedent_estimate::joint_design::compiled::JointDesignInference::Frequentist,
    )
}

/// Compile aligned whole-subject bootstrap refits for the same static-regime target.
/// Every failed refit is retained and any failure withholds the joint covariance.
/// # Errors
/// Invalid source, fewer than 99 refits, cancellation or admission failure.
#[allow(clippy::too_many_arguments)]
pub fn compile_and_evidence_from_joint_static_regime_bootstrap(
    design: &CompiledJointStaticRegime,
    outcomes: &[&[f64]],
    confounders: &[Vec<f64>],
    evidence_id: &str,
    population_id: &str,
    regimes: (&str, &str),
    components: Vec<OutcomeComponent>,
    exposure: Exposure,
    replicates: usize,
    seed: u64,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<CompiledRegimeEvidence, EvidenceRefusal> {
    compile_static_regime_with_inference(
        design,
        outcomes,
        confounders,
        evidence_id,
        population_id,
        regimes,
        components,
        exposure,
        ctx,
        antecedent_estimate::joint_design::compiled::JointDesignInference::StaticRegimeBootstrap {
            replicates,
            seed,
        },
    )
}

#[allow(
    clippy::too_many_arguments,
    clippy::needless_pass_by_value,
    reason = "the shared helper accepts owned outcome declarations"
)]
fn compile_static_regime_with_inference(
    design: &CompiledJointStaticRegime,
    outcomes: &[&[f64]],
    confounders: &[Vec<f64>],
    evidence_id: &str,
    population_id: &str,
    regimes: (&str, &str),
    components: Vec<OutcomeComponent>,
    exposure: Exposure,
    ctx: &antecedent_core::ExecutionContext,
    inference: antecedent_estimate::joint_design::compiled::JointDesignInference,
) -> Result<CompiledRegimeEvidence, EvidenceRefusal> {
    use crate::analysis::joint_panel::designs::PreparedNativeJointDesign;
    use antecedent_estimate::joint_design::{
        compiled::JointDesignRecipe, excursion::ExcursionOutcome,
        static_regime::JointStaticRegimeRecipe,
    };
    let n = design.design.unit_ids.len();
    if outcomes.is_empty()
        || components.len() != outcomes.len()
        || confounders.len() != design.design.confounder_ids.len()
        || outcomes.iter().any(|v| v.len() != n)
        || confounders.iter().any(|v| v.len() != n)
    {
        return Err(EvidenceRefusal::Invalid(
            "regime inputs must align before source preparation".into(),
        ));
    }
    let _source_admission =
        admit_temporal_source(ctx, &design.design.unit_ids, outcomes.len(), confounders.len())?;
    let checked = compile_joint_static_regime(design.design.clone())?;
    if checked.identity != design.identity || checked.units != design.units {
        return Err(EvidenceRefusal::Invalid("compiled static-regime design was modified".into()));
    }
    let Exposure::NoBetweenUnitInterference { assumption_id } = exposure else {
        return Err(EvidenceRefusal::Invalid(
            "static regime requires no between-unit interference".into(),
        ));
    };
    if components.len() != outcomes.len() {
        return Err(EvidenceRefusal::Invalid("regime components and outcomes must align".into()));
    }
    let d = &design.design;
    let recipe = JointStaticRegimeRecipe {
        population: population_id.into(),
        treated_regime_id: regimes.0.into(),
        control_regime_id: regimes.1.into(),
        unit_ids: d.unit_ids.clone(),
        periods: d.periods.clone(),
        assignments: d.realized_assignment.clone(),
        confounder_ids: d.confounder_ids.clone(),
        confounders: confounders.to_vec(),
        outcomes: components
            .iter()
            .zip(outcomes)
            .map(|(c, v)| ExcursionOutcome {
                component_name: c.name.clone(),
                variable_id: c.variable_id.clone(),
                units: c.units.clone(),
                values: v.to_vec(),
            })
            .collect(),
        active_path: d.active_path.clone(),
        reference_path: d.reference_path.clone(),
        method: d.method.clone(),
        min_probability: d.min_probability,
        positivity: d.positivity_assumption_id.clone(),
        exchangeability: d.exchangeability_assumption_id.clone(),
        consistency: d.consistency_assumption_id.clone(),
        followup: d.followup_assumption_id.clone(),
        model_assumption: d.model_assumption_id.clone(),
        isolation: assumption_id,
    };
    let prepared =
        PreparedNativeJointDesign::compile(JointDesignRecipe::StaticRegime { recipe }, ctx)
            .map_err(compiled_decision_refusal)?;
    let result = prepared.execute(inference, ctx).map_err(compiled_decision_refusal)?;
    evidence_from_compiled_static_regime(result, evidence_id, regimes)
}

/// Preserve operational cancellation and admission refusals from a compiled joint design.
#[must_use]
pub fn compiled_decision_refusal(
    error: crate::analysis::joint_panel::JointPanelError,
) -> EvidenceRefusal {
    use crate::analysis::joint_panel::JointPanelError;
    use antecedent_estimate::joint_panel::PanelError;
    match error {
        JointPanelError::Engine(PanelError::Cancelled) => EvidenceRefusal::Cancelled,
        JointPanelError::Engine(PanelError::Resource(message)) => {
            EvidenceRefusal::Resource(message)
        }
        other => EvidenceRefusal::Invalid(other.to_string()),
    }
}

fn admit_temporal_source(
    ctx: &antecedent_core::ExecutionContext,
    unit_ids: &[String],
    outcomes: usize,
    confounders: usize,
) -> Result<antecedent_core::execution::AdmissionLease, EvidenceRefusal> {
    let width = outcomes
        .checked_add(confounders)
        .and_then(|v| v.checked_mul(16))
        .and_then(|v| v.checked_add(128))
        .ok_or_else(|| EvidenceRefusal::Resource("temporal source quote overflow".into()))?;
    let bytes = unit_ids
        .len()
        .checked_mul(width)
        .and_then(|v| {
            unit_ids.iter().try_fold(v, |sum, id| sum.checked_add(id.len().saturating_mul(4)))
        })
        .and_then(|v| u64::try_from(v).ok())
        .ok_or_else(|| EvidenceRefusal::Resource("temporal source quote overflow".into()))?;
    ctx.acquire_admission(0, bytes).map_err(|error| match error {
        antecedent_core::execution::AdmissionError::Cancelled => EvidenceRefusal::Cancelled,
        other => EvidenceRefusal::Resource(other.to_string()),
    })
}

/// Joint experimental evidence retaining the native compiler result and admission.
#[derive(Clone, Debug)]
pub struct CompiledExperimentalEvidence {
    source: crate::analysis::joint_panel::designs::NativeJointDesignResult,
    evidence: JointExperimentEvidence,
}
impl CompiledExperimentalEvidence {
    /// Retained compiler result and source dependency.
    #[must_use]
    pub fn source(&self) -> &crate::analysis::joint_panel::designs::NativeJointDesignResult {
        &self.source
    }
    /// Historical numerical component; current source authorization is separate.
    #[must_use]
    pub fn numerical_evidence(&self) -> &JointExperimentEvidence {
        &self.evidence
    }
}
fn experimental_exposure(
    exposure: Exposure,
) -> Result<antecedent_estimate::joint_design::experiment::ExperimentExposure, EvidenceRefusal> {
    use antecedent_estimate::joint_design::experiment::ExperimentExposure as E;
    Ok(match exposure {
        Exposure::NoBetweenUnitInterference { assumption_id } => {
            E::NoInterference { assumption: assumption_id }
        }
        Exposure::BoundedSpillover { assumption_id, bias_bounds } => {
            E::ContrastBounds { assumption: assumption_id, bounds: bias_bounds }
        }
        Exposure::BoundedArmSpillover {
            assumption_id,
            treated_bias_bounds,
            control_bias_bounds,
        } => E::ArmBounds {
            assumption: assumption_id,
            treated: treated_bias_bounds,
            control: control_bias_bounds,
        },
        Exposure::Unspecified => return Err(EvidenceRefusal::InterferenceUnspecified),
    })
}
/// Bind a retained native matched-pair or switchback source to decision evidence.
/// # Errors
/// Refuses a different source kind, scope or malformed joint law.
#[allow(
    clippy::too_many_lines,
    reason = "the conversion checks the full retained experiment contract"
)]
pub fn evidence_from_compiled_experiment(
    source: crate::analysis::joint_panel::designs::NativeJointDesignResult,
    evidence_id: &str,
    regimes: (&str, &str),
    horizon: u32,
) -> Result<CompiledExperimentalEvidence, EvidenceRefusal> {
    use antecedent_estimate::joint_design::{
        compiled::{JointDesignRecipe, JointNativeDesignLaw},
        experiment::{ExperimentExposure, ExperimentalDesign},
    };
    let JointDesignRecipe::DecisionExperiment { recipe } = source.prepared().recipe() else {
        return Err(EvidenceRefusal::Invalid("compiled source is not a joint experiment".into()));
    };
    let JointNativeDesignLaw::DecisionExperiment(law) = source.law() else {
        return Err(EvidenceRefusal::Invalid("compiled experiment has an incompatible law".into()));
    };
    if regimes != (recipe.treated_regime_id.as_str(), recipe.control_regime_id.as_str())
        || horizon != recipe.horizon
    {
        return Err(EvidenceRefusal::Invalid(
            "experimental intervention or horizon differs from its source".into(),
        ));
    }
    let k = recipe.outcomes.len();
    let width = k
        .checked_mul(2)
        .ok_or_else(|| EvidenceRefusal::Invalid("joint coordinate overflow".into()))?;
    if law.means.len() != width || width.checked_mul(width) != Some(law.covariance.len()) {
        return Err(EvidenceRefusal::Invalid("experimental coordinates are misaligned".into()));
    }
    let exposure = match &recipe.exposure {
        ExperimentExposure::NoInterference { assumption } => {
            Exposure::NoBetweenUnitInterference { assumption_id: assumption.clone() }
        }
        ExperimentExposure::ContrastBounds { assumption, bounds } => Exposure::BoundedSpillover {
            assumption_id: assumption.clone(),
            bias_bounds: bounds.clone(),
        },
        ExperimentExposure::ArmBounds { assumption, treated, control } => {
            Exposure::BoundedArmSpillover {
                assumption_id: assumption.clone(),
                treated_bias_bounds: treated.clone(),
                control_bias_bounds: control.clone(),
            }
        }
    };
    let (kind, details) = match &recipe.design {
        ExperimentalDesign::MatchedPairs {
            cluster_ids,
            pairs,
            matching_variables,
            treatment_probabilities,
            cluster_sizes,
            preassignment,
            independent_pairs,
        } => (
            "matched_pairs",
            DesignDetails::MatchedPairs {
                cluster_ids: cluster_ids.clone(),
                pairs: pairs.clone(),
                matching_variables: matching_variables.clone(),
                treatment_probabilities: treatment_probabilities.clone(),
                cluster_sizes: cluster_sizes.clone(),
                realized_assignment: recipe.assignments.clone(),
                preassignment_assumption_id: preassignment.clone(),
                independent_pairs_assumption_id: independent_pairs.clone(),
                consistency_assumption_id: recipe.consistency.clone(),
            },
        ),
        ExperimentalDesign::Switchback {
            sequence_ids,
            period_ids,
            treatment_probabilities,
            no_carryover,
            washout_rule,
            independent_sequences,
        } => (
            "switchback",
            DesignDetails::Switchback {
                sequence_ids: sequence_ids.clone(),
                period_ids: period_ids.clone(),
                treatment_probabilities: treatment_probabilities.clone(),
                realized_assignment: recipe.assignments.clone(),
                no_carryover_assumption_id: no_carryover.clone(),
                washout_rule_id: washout_rule.clone(),
                washout_periods: 0,
                independent_sequences_assumption_id: independent_sequences.clone(),
                consistency_assumption_id: recipe.consistency.clone(),
            },
        ),
    };
    let components = recipe
        .outcomes
        .iter()
        .map(|o| OutcomeComponent {
            name: o.component_name.clone(),
            variable_id: o.variable_id.clone(),
            units: o.units.clone(),
        })
        .collect();
    let mut evidence = JointExperimentEvidence::new(
        evidence_id,
        &recipe.population,
        regimes,
        horizon,
        components,
        (law.means[..k].to_vec(), law.means[k..].to_vec()),
        Dependence::JointArms { covariance: law.covariance.clone() },
        DesignRecord {
            assignment_design: kind.into(),
            treated_units: law.independent_units,
            control_units: law.independent_units,
            outcome_rows: law.outcome_rows,
        },
        vec![],
        exposure,
    )?;
    evidence.design_contract_id = Some(source.dependency_identity().to_hex());
    evidence.design_details = Some(details);
    Ok(CompiledExperimentalEvidence { source, evidence })
}

#[allow(clippy::too_many_arguments)]
#[allow(
    clippy::needless_pass_by_value,
    reason = "the shared helper accepts owned outcome declarations"
)]
fn compile_decision_experiment(
    design: antecedent_estimate::joint_design::experiment::ExperimentalDesign,
    consistency: &str,
    outcomes: &[&[f64]],
    assigned: &[bool],
    evidence_id: &str,
    population_id: &str,
    regimes: (&str, &str),
    horizon: u32,
    components: Vec<OutcomeComponent>,
    exposure: Exposure,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<CompiledExperimentalEvidence, EvidenceRefusal> {
    use crate::analysis::joint_panel::designs::PreparedNativeJointDesign;
    use antecedent_estimate::joint_design::{
        compiled::{JointDesignInference, JointDesignRecipe},
        excursion::ExcursionOutcome,
        experiment::JointExperimentalRecipe,
    };
    if components.len() != outcomes.len() {
        return Err(EvidenceRefusal::Invalid(
            "experimental components and outcomes must align".into(),
        ));
    }
    let recipe = JointExperimentalRecipe {
        population: population_id.into(),
        treated_regime_id: regimes.0.into(),
        control_regime_id: regimes.1.into(),
        horizon,
        consistency: consistency.into(),
        exposure: experimental_exposure(exposure)?,
        design,
        assignments: assigned.to_vec(),
        outcomes: components
            .iter()
            .zip(outcomes)
            .map(|(c, v)| ExcursionOutcome {
                component_name: c.name.clone(),
                variable_id: c.variable_id.clone(),
                units: c.units.clone(),
                values: v.to_vec(),
            })
            .collect(),
    };
    let prepared =
        PreparedNativeJointDesign::compile(JointDesignRecipe::DecisionExperiment { recipe }, ctx)
            .map_err(compiled_decision_refusal)?;
    let result = prepared
        .execute(JointDesignInference::Frequentist, ctx)
        .map_err(compiled_decision_refusal)?;
    evidence_from_compiled_experiment(result, evidence_id, regimes, horizon)
}

/// Compile and execute retained matched-pair evidence under the caller's execution context.
/// # Errors
/// Invalid design, source, scientific assumptions, cancellation or resource refusal.
#[allow(clippy::too_many_arguments)]
pub fn compile_and_evidence_from_matched_pairs(
    design: &CompiledMatchedPairDesign,
    outcomes: &[&[f64]],
    assigned: &[bool],
    evidence_id: &str,
    population_id: &str,
    regimes: (&str, &str),
    horizon: u32,
    components: Vec<OutcomeComponent>,
    exposure: Exposure,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<CompiledExperimentalEvidence, EvidenceRefusal> {
    use antecedent_estimate::joint_design::experiment::ExperimentalDesign;
    let n = design.design.cluster_ids.len();
    if assigned.len() != n
        || outcomes.is_empty()
        || components.len() != outcomes.len()
        || outcomes.iter().any(|v| v.len() != n)
    {
        return Err(EvidenceRefusal::Invalid(
            "paired inputs must align before source preparation".into(),
        ));
    }
    let _admission = admit_temporal_source(ctx, &design.design.cluster_ids, outcomes.len(), 0)?;
    let checked = compile_matched_pair_design(design.design.clone())?;
    if checked.identity != design.identity {
        return Err(EvidenceRefusal::Invalid("compiled paired design was modified".into()));
    }
    let d = &design.design;
    compile_decision_experiment(
        ExperimentalDesign::MatchedPairs {
            cluster_ids: d.cluster_ids.clone(),
            pairs: d.pairs.clone(),
            matching_variables: d.matching_variables.clone(),
            treatment_probabilities: d.treatment_probabilities.clone(),
            cluster_sizes: d.cluster_sizes.clone(),
            preassignment: d.preassignment_assumption_id.clone(),
            independent_pairs: d.independent_pairs_assumption_id.clone(),
        },
        &d.consistency_assumption_id,
        outcomes,
        assigned,
        evidence_id,
        population_id,
        regimes,
        horizon,
        components,
        exposure,
        ctx,
    )
}

/// Compile and execute retained zero-washout switchback evidence.
/// # Errors
/// Invalid design, source, scientific assumptions, cancellation or resource refusal.
#[allow(clippy::too_many_arguments)]
pub fn compile_and_evidence_from_joint_switchback(
    design: &CompiledJointSwitchback,
    outcomes: &[&[f64]],
    assigned: &[bool],
    evidence_id: &str,
    population_id: &str,
    regimes: (&str, &str),
    horizon: u32,
    components: Vec<OutcomeComponent>,
    exposure: Exposure,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<CompiledExperimentalEvidence, EvidenceRefusal> {
    use antecedent_estimate::joint_design::experiment::ExperimentalDesign;
    let n = design.design.sequence_ids.len();
    if assigned.len() != n
        || outcomes.is_empty()
        || components.len() != outcomes.len()
        || outcomes.iter().any(|v| v.len() != n)
    {
        return Err(EvidenceRefusal::Invalid(
            "switchback inputs must align before source preparation".into(),
        ));
    }
    let _admission = admit_temporal_source(ctx, &design.design.sequence_ids, outcomes.len(), 0)?;
    let checked = compile_joint_switchback(design.design.clone())?;
    if checked.identity != design.identity {
        return Err(EvidenceRefusal::Invalid("compiled switchback design was modified".into()));
    }
    let d = &design.design;
    compile_decision_experiment(
        ExperimentalDesign::Switchback {
            sequence_ids: d.sequence_ids.clone(),
            period_ids: d.period_ids.clone(),
            treatment_probabilities: d.treatment_probabilities.clone(),
            no_carryover: d.no_carryover_assumption_id.clone(),
            washout_rule: d.washout_rule_id.clone(),
            independent_sequences: d.independent_sequences_assumption_id.clone(),
        },
        &d.consistency_assumption_id,
        outcomes,
        assigned,
        evidence_id,
        population_id,
        regimes,
        horizon,
        components,
        exposure,
        ctx,
    )
}

/// Cluster decision evidence retaining its compiler result and resource admission.
#[derive(Clone, Debug)]
pub struct CompiledClusterEvidence {
    source: crate::analysis::joint_panel::designs::NativeJointDesignResult,
    evidence: JointExperimentEvidence,
}
impl CompiledClusterEvidence {
    /// Retained compiled source and dependency identity.
    #[must_use]
    pub fn source(&self) -> &crate::analysis::joint_panel::designs::NativeJointDesignResult {
        &self.source
    }
    /// Historical numerical component; current authorization is separate.
    #[must_use]
    pub fn numerical_evidence(&self) -> &JointExperimentEvidence {
        &self.evidence
    }
}

/// Bind a retained cluster design result to decision evidence without asserting cross-arm covariance.
/// # Errors
/// Refuses a different native source, target or malformed law.
pub fn evidence_from_compiled_cluster(
    source: crate::analysis::joint_panel::designs::NativeJointDesignResult,
    evidence_id: &str,
    regimes: (&str, &str),
    horizon: u32,
) -> Result<CompiledClusterEvidence, EvidenceRefusal> {
    use antecedent_estimate::joint_design::{
        compiled::{JointDesignRecipe, JointNativeDesignLaw},
        experiment::ExperimentExposure,
    };
    let JointDesignRecipe::DecisionCluster { recipe } = source.prepared().recipe() else {
        return Err(EvidenceRefusal::Invalid("compiled source is not a cluster experiment".into()));
    };
    let JointNativeDesignLaw::DecisionCluster(law) = source.law() else {
        return Err(EvidenceRefusal::Invalid(
            "compiled cluster source has an incompatible law".into(),
        ));
    };
    if regimes != (recipe.treated_regime_id.as_str(), recipe.control_regime_id.as_str())
        || horizon != recipe.horizon
    {
        return Err(EvidenceRefusal::Invalid(
            "cluster intervention or horizon differs from its source".into(),
        ));
    }
    let exposure = match &recipe.exposure {
        ExperimentExposure::NoInterference { assumption } => {
            Exposure::NoBetweenUnitInterference { assumption_id: assumption.clone() }
        }
        ExperimentExposure::ContrastBounds { assumption, bounds } => Exposure::BoundedSpillover {
            assumption_id: assumption.clone(),
            bias_bounds: bounds.clone(),
        },
        ExperimentExposure::ArmBounds { assumption, treated, control } => {
            Exposure::BoundedArmSpillover {
                assumption_id: assumption.clone(),
                treated_bias_bounds: treated.clone(),
                control_bias_bounds: control.clone(),
            }
        }
    };
    let components = recipe
        .outcomes
        .iter()
        .map(|o| OutcomeComponent {
            name: o.component_name.clone(),
            variable_id: o.variable_id.clone(),
            units: o.units.clone(),
        })
        .collect();
    let adjustments = recipe
        .adjustments
        .iter()
        .map(|a| ComponentAdjustment {
            component: a.component_name.clone(),
            kind: if a.centered {
                AdjustmentKind::CenteredFixedCuped {
                    covariate: a.covariate.clone(),
                    coefficient: a.coefficient,
                }
            } else {
                AdjustmentKind::FixedCuped {
                    covariate: a.covariate.clone(),
                    coefficient: a.coefficient,
                }
            },
        })
        .collect();
    let mut evidence = JointExperimentEvidence::new(
        evidence_id,
        &recipe.population,
        regimes,
        horizon,
        components,
        (law.treated_means.clone(), law.control_means.clone()),
        Dependence::Joint {
            treated: law.treated_covariance.clone(),
            control: law.control_covariance.clone(),
        },
        DesignRecord {
            assignment_design: "cluster".into(),
            treated_units: law.treated_clusters,
            control_units: law.control_clusters,
            outcome_rows: law.outcome_rows,
        },
        adjustments,
        exposure,
    )?;
    evidence.design_contract_id = Some(source.dependency_identity().to_hex());
    Ok(CompiledClusterEvidence { source, evidence })
}

/// Compile a complete cluster source, compute its declared adjustments and retain its result.
/// # Errors
/// Invalid or unsupported source, scientific premises, cancellation or resource admission.
pub fn compile_and_evidence_from_cluster(
    recipe: antecedent_estimate::joint_design::cluster_decision::JointClusterRecipe,
    evidence_id: &str,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<CompiledClusterEvidence, EvidenceRefusal> {
    use crate::analysis::joint_panel::designs::PreparedNativeJointDesign;
    use antecedent_estimate::joint_design::compiled::{JointDesignInference, JointDesignRecipe};
    let _admission = admit_temporal_source(
        ctx,
        &recipe.cluster_ids,
        recipe.outcomes.len(),
        recipe.adjustments.len(),
    )?;
    let treated = recipe.treated_regime_id.clone();
    let control = recipe.control_regime_id.clone();
    let horizon = recipe.horizon;
    let prepared =
        PreparedNativeJointDesign::compile(JointDesignRecipe::DecisionCluster { recipe }, ctx)
            .map_err(compiled_decision_refusal)?;
    let result = prepared
        .execute(JointDesignInference::Frequentist, ctx)
        .map_err(compiled_decision_refusal)?;
    evidence_from_compiled_cluster(result, evidence_id, (&treated, &control), horizon)
}
