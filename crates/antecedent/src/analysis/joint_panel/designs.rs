//! Design-native compiled joint functionals, independent of the nonlinear panel fitter.
// SPDX-License-Identifier: MIT OR Apache-2.0
mod native;
use super::{JointPanelError, contracts};
use antecedent_contract::{QuantityRole, ScientificQuantity};
use antecedent_core::{
    CanonicalEncode, CanonicalWriter, ExecutionContext, ScientificTarget, SemanticDigest,
    TargetForm, VariableId,
};
use antecedent_design::scientific_source::{
    ScientificDecisionSource, ScientificProjectionBasis, ScientificProjectionOutput,
    ScientificSourcePayload, ScientificSourceProjection,
};
use antecedent_estimate::joint_design::compiled::{
    JointDesignInference, JointDesignLaw, JointDesignRecipe,
};
use antecedent_expr::{
    CausalExprArena, DerivationMeta, DomainRef, ExprNode, IdentifiedEstimand, OutcomeExprId,
};
use antecedent_identify::compiled_product::{IdentificationObligation, RequiredFactor};
use antecedent_identify::{
    CompiledIdentificationProduct, CompiledProductParts, DerivationTrace, IdentificationClaimClass,
    IdentificationOutcome,
};
use antecedent_io::distribution_artifact::{
    DistributionArtifact, DistributionCalibration, DistributionIdentity, DistributionMetadata,
    DistributionProvenance, DistributionTrust, DrawAlignment,
};
use antecedent_io::quantity_wire::DistributionMeaningWire;
pub(crate) use native::replay_native_joint_design_execution;
pub use native::{
    NativeJointDesignResult, NativeJointIdentification, PreparedNativeJointDesign,
    replay_native_joint_design, replay_native_joint_design_with_limits,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Complete program key mapped to one exact scientific target world.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JointDesignWorldBinding {
    /// Key retained in the design assignment/longitudinal contract.
    pub key: String,
    /// Exact scientific world, including its intervention/history/population semantics.
    pub world: u32,
    /// Exact source-codebook scientific world content, including interventions,
    /// population, initial history and time frame. A matching label is insufficient.
    pub target_identity: String,
}
/// Semantic identity of one source-codebook world, including exact schedules,
/// initial histories, population, sampling unit and outcome definition.
#[must_use]
pub fn joint_design_world_identity(world: &antecedent_core::InterventionWorld) -> SemanticDigest {
    let mut writer = CanonicalWriter::new("antecedent.joint_design.source_world.v1");
    world.encode(&mut writer);
    SemanticDigest::from_bytes(writer.finish())
}
/// Physical simultaneous assignments represented by one longitudinal joint-arm code.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JointDesignArm {
    /// Joint-arm code in every recorded episode and requested schedule.
    pub arm: usize,
    /// Explicit variable and finite hard-setting value, sorted by variable ID.
    pub levels: Vec<(u32, f64)>,
}
/// Original source measurement and sampling contract. This is a scientific
/// premise about how the recorded outcomes were obtained, not inferred from labels.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JointDesignSourceBinding {
    /// Actual measured outcome column.
    pub outcome_variable: u32,
    /// Actual measurement units.
    pub outcome_units: String,
    /// Independent source sampling/assignment unit.
    pub sampling_unit: String,
    /// Actual source population, before any requested transport.
    pub population: antecedent_io::TargetPopulationWire,
    /// Recorded terminal-mean horizon; zero denotes a static decision outcome.
    pub outcome_horizon: u32,
    /// Original population/sampling-frame declaration retained by the method owner.
    pub population_description: String,
}
impl JointDesignSourceBinding {
    /// Declare an original source measurement contract using an explicit source world.
    /// A later target with changed population, horizon or outcome will be refused.
    /// # Errors
    /// A registry-backed population cannot be encoded without its source registry.
    pub fn from_world(
        world: &antecedent_core::InterventionWorld,
        population_description: String,
    ) -> Result<Self, JointPanelError> {
        Ok(Self {
            outcome_variable: world.binding().outcome().outcome().raw(),
            outcome_units: world.binding().outcome().units().name().into(),
            sampling_unit: world.binding().unit().name().into(),
            population: antecedent_io::TargetPopulationWire::from_domain(
                world.binding().population(),
            )
            .map_err(failure)?,
            outcome_horizon: world.time().horizon_steps().unwrap_or(0),
            population_description,
        })
    }
}
/// Versioned executable design proof, retained across both statistical modes.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JointDesignProof {
    /// Version of the typed proof representation.
    pub version: u16,
    /// Actual identifying functional and its required assignment/history/rank/cohort support.
    pub recipe: JointDesignRecipe,
    /// Actual source population, outcome aggregation and independent sampling unit.
    pub source_binding: JointDesignSourceBinding,
    /// Exact scientific world binding in design-functional output order.
    pub worlds: Vec<JointDesignWorldBinding>,
    /// Required physical arm dictionary for longitudinal schedules; randomized
    /// whole-program arms instead bind directly to full source-codebook worlds.
    #[serde(default)]
    pub longitudinal_arms: Vec<JointDesignArm>,
}
/// Compiled design owner retaining source data, identifying functional and explicit assumptions.
#[derive(Clone, Debug)]
pub struct PreparedJointDesign {
    target: Arc<ScientificTarget>,
    proof: Arc<JointDesignProof>,
    identification: Arc<CompiledIdentificationProduct>,
    dependency: SemanticDigest,
    point: Arc<JointDesignLaw>,
    _lease: Arc<antecedent_core::execution::AdmissionLease>,
}
/// Executed native design law; the original proof and target survive downstream ports.
#[derive(Clone, Debug)]
pub struct JointDesignResult {
    prepared: PreparedJointDesign,
    inference: Arc<JointDesignInference>,
    law: Arc<JointDesignLaw>,
    dependency: SemanticDigest,
    admission: Arc<antecedent_core::execution::AdmissionLease>,
}
fn failure(error: impl std::fmt::Display) -> JointPanelError {
    JointPanelError::compile(error.to_string())
}
fn check_cancel(ctx: &ExecutionContext) -> Result<(), JointPanelError> {
    if ctx.cancellation.is_cancelled() {
        Err(JointPanelError::Engine(super::engine::PanelError::Cancelled))
    } else {
        Ok(())
    }
}
fn identity(
    target: &ScientificTarget,
    proof: &JointDesignProof,
) -> Result<SemanticDigest, JointPanelError> {
    let encoded = antecedent_io::convert::to_cbor(proof).map_err(failure)?;
    let mut writer = CanonicalWriter::new("antecedent.compiled_joint_design.v1");
    writer.bytes(target.identity().as_bytes()).bytes(&encoded);
    Ok(SemanticDigest::from_bytes(writer.finish()))
}
pub(super) fn retained_law_identity<T: serde::Serialize>(
    source: SemanticDigest,
    inference: &JointDesignInference,
    law: &T,
) -> Result<SemanticDigest, JointPanelError> {
    let mut writer = CanonicalWriter::new("antecedent.joint_design.fitted_law.v1");
    writer
        .bytes(source.as_bytes())
        .bytes(&antecedent_io::convert::to_cbor(inference).map_err(failure)?)
        .bytes(&antecedent_io::convert::to_cbor(law).map_err(failure)?);
    Ok(SemanticDigest::from_bytes(writer.finish()))
}
fn execution_failure(ctx: &ExecutionContext, error: impl std::fmt::Display) -> JointPanelError {
    if ctx.cancellation.is_cancelled() {
        JointPanelError::Engine(super::engine::PanelError::Cancelled)
    } else {
        failure(error)
    }
}
fn execute_world_recipe(
    recipe: &JointDesignRecipe,
    inference: &JointDesignInference,
    ctx: &ExecutionContext,
) -> Result<JointDesignLaw, JointPanelError> {
    match recipe.execute_native(inference, ctx).map_err(|error| execution_failure(ctx, error))? {
        antecedent_estimate::joint_design::compiled::JointNativeDesignLaw::Moments(law) => Ok(law),
        _ => Err(failure("method-native target is not a joint world-mean law")),
    }
}
impl PreparedJointDesign {
    /// Compile exact design-native means. This route never relabels a Gaussian panel
    /// estimator as g-computation, IV or a rollout estimator.
    ///
    /// # Errors
    /// Unsupported proof version, mismatched world binding, absent design premises,
    /// incomplete support, cancellation or admission refusal.
    #[allow(
        clippy::too_many_lines,
        reason = "one atomic proof validates source, physical worlds and identifying expression before constructing authority"
    )]
    pub fn compile(
        target: ScientificTarget,
        proof: JointDesignProof,
        ctx: &ExecutionContext,
    ) -> Result<Self, JointPanelError> {
        check_cancel(ctx)?;
        if proof.version != 1 {
            return Err(failure("unsupported joint-design proof version"));
        }
        // This port is for joint world means. Native IV and DiD owners retain their
        // structural/cohort contrast target and must not be silently cast to world means.
        if !matches!(
            proof.recipe,
            JointDesignRecipe::Randomized { .. }
                | JointDesignRecipe::Longitudinal { .. }
                | JointDesignRecipe::GaussianLongitudinal { .. }
        ) {
            return Err(failure(
                "this scientific world-mean port requires a mean-functional design; use the native structural/cohort target owner for IV or DiD",
            ));
        }
        let worlds = contracts::target_worlds(&target)?;
        if worlds.len() != proof.worlds.len()
            || proof.worlds.iter().enumerate().any(|(i, b)| {
                b.world != worlds[i].id().raw()
                    || b.target_identity != joint_design_world_identity(&worlds[i]).to_hex()
            })
        {
            return Err(failure(
                "joint-design bindings must cover each exact target world in retained order",
            ));
        }
        if worlds.iter().any(|w| {
            !matches!(w.binding().outcome().functional(), antecedent_core::OutcomeFunctional::Mean)
        }) {
            return Err(failure(
                "native joint-design world means do not establish another outcome functional",
            ));
        }
        let source_population = proof.source_binding.population.to_domain().map_err(failure)?;
        if proof.source_binding.population_description.trim().is_empty()
            || worlds.iter().any(|w| {
                w.binding().population() != &source_population
                    || w.binding().unit().name() != proof.source_binding.sampling_unit
                    || w.binding().outcome().outcome().raw()
                        != proof.source_binding.outcome_variable
                    || w.binding().outcome().units().name() != proof.source_binding.outcome_units
                    || w.time().horizon_steps().unwrap_or(0) != proof.source_binding.outcome_horizon
                    || !w.binding().conditioning().variables().is_empty()
            })
        {
            return Err(failure(
                "joint-design target changes the source population, sampling unit, outcome measurement, terminal aggregation horizon or conditioning without a supported translation",
            ));
        }
        match &proof.recipe {
            JointDesignRecipe::Randomized { design, .. }
                if design.sampling_unit != proof.source_binding.sampling_unit
                    || design.population != proof.source_binding.population_description =>
            {
                return Err(failure(
                    "randomization contract differs from the original source sampling frame",
                ));
            }
            JointDesignRecipe::Longitudinal { design, .. }
                if design.sampling_population != proof.source_binding.population_description =>
            {
                return Err(failure(
                    "longitudinal history contract differs from the original source sampling population",
                ));
            }
            JointDesignRecipe::GaussianLongitudinal { design, .. }
                if design.population != proof.source_binding.population_description =>
            {
                return Err(failure(
                    "continuous longitudinal model differs from the original source sampling population",
                ));
            }
            _ => {}
        }
        if let JointDesignRecipe::Longitudinal { design, .. } = &proof.recipe {
            use antecedent_core::{
                Intervention, InterventionSequence, SequencedIntervention, TemporalPolicy, Value,
                WorldAction,
            };
            let arms = &proof.longitudinal_arms;
            // The dictionary's size is the proof's own; the admission below charges for it.
            let mut seen_arms = std::collections::BTreeSet::new();
            if arms.is_empty()
                || arms.iter().any(|a| {
                    a.levels.is_empty()
                        || !seen_arms.insert(a.arm)
                        || a.levels.iter().any(|(_, v)| !v.is_finite())
                        || a.levels.windows(2).any(|w| w[0].0 >= w[1].0)
                })
            {
                return Err(failure(
                    "longitudinal proof requires a unique complete physical joint-arm dictionary",
                ));
            }
            let axis = arms[0].levels.iter().map(|(v, _)| *v).collect::<Vec<_>>();
            if arms.iter().any(|a| a.levels.iter().map(|(v, _)| *v).collect::<Vec<_>>() != axis) {
                return Err(failure(
                    "every joint arm must set the same complete intervention coordinate axis",
                ));
            }
            for ((_, schedule), world) in design.worlds.iter().zip(worlds) {
                let actions=schedule.iter().map(|code|arms.iter().find(|a|a.arm==*code).ok_or_else(||failure("requested longitudinal schedule uses an undefined physical joint arm"))).collect::<Result<Vec<_>,_>>()?;
                let expected =
                    if world.time().is_static() && schedule.len() == 1 {
                        WorldAction::at_decision(actions[0].levels.iter().map(|(v, x)| {
                            Intervention::set(VariableId::from_raw(*v), Value::f64(*x))
                        }))
                        .map_err(failure)?
                    } else {
                        if world.time().horizon_steps() != u32::try_from(schedule.len()).ok() {
                            return Err(failure(
                                "longitudinal schedule horizon differs from scientific world",
                            ));
                        }
                        let mut steps = Vec::new();
                        for (t, arm) in actions.iter().enumerate() {
                            let at = i32::try_from(t).map_err(failure)?;
                            for (v, x) in &arm.levels {
                                steps.push(SequencedIntervention::new(
                                    Intervention::set(VariableId::from_raw(*v), Value::f64(*x)),
                                    TemporalPolicy::pulse(at),
                                ));
                            }
                        }
                        WorldAction::schedule(
                            Intervention::Sequence(InterventionSequence::new(steps)),
                            TemporalPolicy::sustained(
                                0,
                                i32::try_from(schedule.len() - 1).map_err(failure)?,
                            ),
                        )
                        .map_err(failure)?
                    };
                if world.action() != &expected {
                    return Err(failure(
                        "longitudinal arm history does not equal the actual scientific intervention schedule",
                    ));
                }
            }
        } else if !proof.longitudinal_arms.is_empty() {
            return Err(failure(
                "a non-longitudinal proof may not silently introduce a second arm dictionary",
            ));
        }
        if let JointDesignRecipe::GaussianLongitudinal { design, .. } = &proof.recipe {
            use antecedent_core::{
                Intervention, InterventionSequence, SequencedIntervention, TemporalPolicy, Value,
                WorldAction,
            };
            for (regime, world) in design.worlds.iter().zip(worlds) {
                let expected = if world.time().is_static() && regime.treatments.len() == 1 {
                    WorldAction::at_decision(
                        design.channel_variables.iter().zip(&regime.treatments[0]).map(
                            |(variable, value)| {
                                Intervention::set(
                                    VariableId::from_raw(*variable),
                                    Value::f64(*value),
                                )
                            },
                        ),
                    )
                    .map_err(failure)?
                } else {
                    if world.time().horizon_steps() != u32::try_from(regime.treatments.len()).ok() {
                        return Err(failure(
                            "continuous g-formula horizon differs from its actual scientific world",
                        ));
                    }
                    let mut steps = Vec::new();
                    for (t, levels) in regime.treatments.iter().enumerate() {
                        for (variable, value) in design.channel_variables.iter().zip(levels) {
                            steps.push(SequencedIntervention::new(
                                Intervention::set(
                                    VariableId::from_raw(*variable),
                                    Value::f64(*value),
                                ),
                                TemporalPolicy::pulse(i32::try_from(t).map_err(failure)?),
                            ));
                        }
                    }
                    WorldAction::schedule(
                        Intervention::Sequence(InterventionSequence::new(steps)),
                        TemporalPolicy::sustained(
                            0,
                            i32::try_from(regime.treatments.len() - 1).map_err(failure)?,
                        ),
                    )
                    .map_err(failure)?
                };
                if world.action() != &expected {
                    return Err(failure(
                        "continuous treatment history does not equal the target's actual physical intervention schedule",
                    ));
                }
            }
        }
        let quote = antecedent_io::convert::to_cbor(&proof).map_err(failure)?.len();
        let bytes = u64::try_from(quote)
            .ok()
            .and_then(|v| v.checked_mul(128))
            .ok_or_else(|| failure("joint-design admission size overflow"))?;
        let lease = native::admitted(ctx, bytes)?;
        let point = execute_world_recipe(&proof.recipe, &JointDesignInference::Frequentist, ctx)?;
        if point.estimate().worlds.iter().zip(&proof.worlds).any(|(key, b)| key != &b.key)
            || point.estimate().worlds.len() != proof.worlds.len()
        {
            return Err(failure(
                "assignment/history program keys do not match exact scientific world bindings",
            ));
        }
        check_cancel(ctx)?;
        let mut arena = CausalExprArena::new();
        let mut estimands = Vec::new();
        let mut trace = DerivationTrace::default();
        // Derived observable columns are deterministic transformations retained by
        // the recipe: HT scores/weights, or time-indexed complete-history states.
        let mut source_variables =
            worlds.iter().map(|w| w.binding().outcome().outcome().raw()).collect::<Vec<_>>();
        for world in worlds {
            source_variables
                .extend(world.binding().conditioning().variables().iter().map(|v| v.raw()));
            if let Some(actions) = world.action().at_decision_interventions() {
                source_variables.extend(
                    actions
                        .iter()
                        .flat_map(antecedent_core::Intervention::target_variables)
                        .map(VariableId::raw),
                );
            }
            if let Some((action, _)) = world.action().schedule_parts() {
                source_variables.extend(action.target_variables().iter().map(|v| v.raw()));
            }
        }
        for arm in &proof.longitudinal_arms {
            source_variables.extend(arm.levels.iter().map(|(v, _)| *v));
        }
        if let JointDesignRecipe::GaussianLongitudinal { design, .. } = &proof.recipe {
            source_variables.extend(
                design.channel_variables.iter().chain(&design.covariate_variables).copied(),
            );
        }
        let max_raw = source_variables.into_iter().max().unwrap_or(0);
        let synthetic = |offset: u32| {
            max_raw
                .checked_add(offset)
                .map(VariableId::from_raw)
                .ok_or_else(|| failure("derived identifying-functional variable IDs overflow"))
        };
        for (world_index, world) in worlds.iter().enumerate() {
            let empty = arena.empty_var_set();
            let no_intervention = arena.empty_intervention_set();
            let functional = match &proof.recipe {
                JointDesignRecipe::Randomized { .. } => {
                    let score = synthetic(u32::try_from(world_index + 2).map_err(failure)?)?;
                    let weight = synthetic(1)?;
                    let vars = arena.intern_var_set([score]);
                    let score_law = arena.intern_distribution(
                        vars,
                        empty,
                        no_intervention,
                        DomainRef::Observational,
                    );
                    let numerator = arena.intern(ExprNode::Expectation {
                        function: OutcomeExprId::identity(score),
                        distribution: score_law,
                    });
                    let vars = arena.intern_var_set([weight]);
                    let weight_law = arena.intern_distribution(
                        vars,
                        empty,
                        no_intervention,
                        DomainRef::Observational,
                    );
                    let denominator = arena.intern(ExprNode::Expectation {
                        function: OutcomeExprId::identity(weight),
                        distribution: weight_law,
                    });
                    arena.intern(ExprNode::Ratio { numerator, denominator })
                }
                JointDesignRecipe::Longitudinal { design, .. } => {
                    let horizon = design.worlds[world_index].1.len();
                    let states = (0..horizon)
                        .map(|t| synthetic(u32::try_from(2 * t + 1).map_err(failure)?))
                        .collect::<Result<Vec<_>, _>>()?;
                    let arms = (0..horizon)
                        .map(|t| synthetic(u32::try_from(2 * t + 2).map_err(failure)?))
                        .collect::<Result<Vec<_>, _>>()?;
                    let mut factors = Vec::new();
                    for t in 0..horizon {
                        let vars = arena.intern_var_set([states[t]]);
                        let conditioned =
                            arena.intern_var_set(states[..t].iter().chain(&arms[..t]).copied());
                        factors.push(arena.intern_distribution(
                            vars,
                            conditioned,
                            no_intervention,
                            DomainRef::Observational,
                        ));
                    }
                    let vars = arena.intern_var_set([world.binding().outcome().outcome()]);
                    let conditioned = arena.intern_var_set(states.iter().chain(&arms).copied());
                    factors.push(arena.intern_distribution(
                        vars,
                        conditioned,
                        no_intervention,
                        DomainRef::Observational,
                    ));
                    let factors = arena.intern_list(factors);
                    let product = arena.intern(ExprNode::Product(factors));
                    let states = arena.intern_var_set(states);
                    let distribution =
                        arena.intern(ExprNode::SumOut { variables: states, expr: product });
                    arena.intern(ExprNode::Expectation {
                        function: OutcomeExprId::identity(world.binding().outcome().outcome()),
                        distribution,
                    })
                }
                JointDesignRecipe::GaussianLongitudinal { design, .. } => {
                    let h = design.worlds[world_index].treatments.len();
                    let d = design.covariate_variables.len();
                    let m = design.channel_variables.len();
                    let states = (0..h * d)
                        .map(|j| synthetic(u32::try_from(j + 1).map_err(failure)?))
                        .collect::<Result<Vec<_>, _>>()?;
                    let arms = (0..h * m)
                        .map(|j| synthetic(u32::try_from(h * d + j + 1).map_err(failure)?))
                        .collect::<Result<Vec<_>, _>>()?;
                    let mut factors = Vec::new();
                    for t in 0..h {
                        for j in 0..d {
                            let at = t * d + j;
                            let vars = arena.intern_var_set([states[at]]);
                            let conditioned = arena
                                .intern_var_set(states[..at].iter().chain(&arms[..t * m]).copied());
                            factors.push(arena.intern_distribution(
                                vars,
                                conditioned,
                                no_intervention,
                                DomainRef::Observational,
                            ));
                        }
                    }
                    let vars = arena.intern_var_set([world.binding().outcome().outcome()]);
                    let conditioned = arena.intern_var_set(states.iter().chain(&arms).copied());
                    factors.push(arena.intern_distribution(
                        vars,
                        conditioned,
                        no_intervention,
                        DomainRef::Observational,
                    ));
                    let factors = arena.intern_list(factors);
                    let product = arena.intern(ExprNode::Product(factors));
                    let states = arena.intern_var_set(states);
                    let distribution =
                        arena.intern(ExprNode::IntegralOut { variables: states, expr: product });
                    arena.intern(ExprNode::Expectation {
                        function: OutcomeExprId::identity(world.binding().outcome().outcome()),
                        distribution,
                    })
                }
                _ => {
                    return Err(failure(
                        "native contrast owner cannot issue a world-mean functional",
                    ));
                }
            };
            arena.set_derivation(
                functional,
                DerivationMeta::rule(
                    proof.recipe.method(),
                    Some(Arc::from(proof.recipe.functional())),
                ),
            );
            estimands.push(IdentifiedEstimand::new(
                proof.recipe.method(),
                Arc::from([]),
                Arc::from([]),
                Arc::from([]),
                functional,
                None,
            ));
        }
        trace.push(proof.recipe.method(), proof.recipe.functional());
        let mut parts = CompiledProductParts::new(
            IdentificationOutcome::Identified,
            IdentificationClaimClass::IdentifiedClaim,
        );
        parts.world_obligations = worlds
            .iter()
            .map(|w| {
                IdentificationObligation::for_world(
                    w.id(),
                    IdentificationOutcome::Identified,
                    IdentificationClaimClass::IdentifiedClaim,
                    vec![],
                    None,
                )
            })
            .collect();
        if matches!(target.form(), TargetForm::Contrast(_) | TargetForm::DoseTime(_)) {
            parts.assembled = Some(IdentificationObligation::assembled(
                IdentificationOutcome::Identified,
                IdentificationClaimClass::IdentifiedClaim,
                vec![],
                None,
            ));
        }
        parts.required_factors.push(RequiredFactor::new(proof.recipe.functional(), Arc::from([])));
        let mut assumptions = target.assumptions().records().clone();
        for (i, description) in
            native::summary(&proof.recipe).required_assumptions.into_iter().enumerate()
        {
            assumptions.push(antecedent_core::AssumptionRecord {
                assumption: antecedent_core::Assumption::Custom {
                    id: format!("{}:premise:{i}", proof.recipe.method()).into(),
                    description: description.into(),
                },
                source: antecedent_core::AssumptionSource::UserDeclared,
                scope: antecedent_core::AssumptionScope::Identification,
                status: antecedent_core::AssumptionStatus::Declared,
            });
        }
        let identification =
            CompiledIdentificationProduct::new(estimands, arena, trace, assumptions, vec![], parts)
                .and_then(|p| p.bind_target(&target))
                .map_err(failure)?;
        let dependency = identity(&target, &proof)?;
        Ok(Self {
            target: Arc::new(target),
            proof: Arc::new(proof),
            identification: Arc::new(identification),
            dependency,
            point: Arc::new(point),
            _lease: lease,
        })
    }
    /// Exact identifying functional and design premises, independent of statistical mode.
    #[must_use]
    pub fn proof(&self) -> &JointDesignProof {
        &self.proof
    }
    /// Common target-bound identification product.
    pub fn identification(&self) -> &CompiledIdentificationProduct {
        &self.identification
    }
    /// Source-data/design/target identity required for current artifact consumption.
    #[must_use]
    pub const fn dependency(&self) -> SemanticDigest {
        self.dependency
    }
    /// Execute native uncertainty without substituting frequentist covariance for a posterior.
    /// # Errors
    /// Incompatible posterior model, missing full-history support, cancellation or admission refusal.
    pub fn execute(
        &self,
        inference: JointDesignInference,
        ctx: &ExecutionContext,
    ) -> Result<JointDesignResult, JointPanelError> {
        check_cancel(ctx)?;
        let draws = match &inference {
            JointDesignInference::Frequentist
            | JointDesignInference::SwitchbackExactRandomization { .. }
            | JointDesignInference::SyntheticConformal { .. } => 1,
            JointDesignInference::LocalComplierBayesian { draws, .. } => *draws,
            JointDesignInference::GaussianLongitudinalBootstrap { replicates, .. } => *replicates,
            JointDesignInference::GaussianLongitudinalBayesian { prior } => prior.draws,
            JointDesignInference::RandomizedGaussian { prior } => prior.draws,
            JointDesignInference::RandomizedFinitePopulationGaussian { prior } => prior.arm.draws,
            JointDesignInference::LongitudinalDirichlet { prior } => prior.draws,
            JointDesignInference::SwitchbackGaussian { prior } => prior.draws,
            JointDesignInference::InstrumentedGaussian { prior } => prior.draws,
            JointDesignInference::RolloutGaussian { prior } => prior.draws,
            JointDesignInference::SyntheticBayesian { prior } => prior.draws,
        };
        let bytes = draws
            .checked_mul(self.proof.worlds.len())
            .and_then(|v| v.checked_mul(32))
            .and_then(|v| u64::try_from(v).ok())
            .ok_or_else(|| failure("joint-design law admission overflow"))?;
        let lease = native::admitted(ctx, bytes)?;
        let law = if matches!(inference, JointDesignInference::Frequentist) {
            self.point.clone()
        } else {
            Arc::new(execute_world_recipe(&self.proof.recipe, &inference, ctx)?)
        };
        let dependency = retained_law_identity(self.dependency, &inference, law.as_ref())?;
        check_cancel(ctx)?;
        Ok(JointDesignResult {
            prepared: self.clone(),
            inference: Arc::new(inference),
            law,
            dependency,
            admission: lease,
        })
    }
}
impl JointDesignResult {
    /// Complete native estimate/posterior with its retained inferential meaning.
    #[must_use]
    pub fn law(&self) -> &JointDesignLaw {
        &self.law
    }
    /// Exact statistical procedure and aligned draw-slot identity, including prior and seed.
    #[must_use]
    pub const fn dependency_identity(&self) -> SemanticDigest {
        self.dependency
    }
    /// Typed original scientific and numerical owner.
    #[must_use]
    pub fn prepared(&self) -> &PreparedJointDesign {
        &self.prepared
    }
    fn coordinates(&self) -> Result<Vec<ScientificQuantity>, JointPanelError> {
        let worlds = contracts::target_worlds(&self.prepared.target)?;
        let quantities = worlds
            .iter()
            .map(|world| ScientificQuantity {
                variable_id: world.binding().outcome().outcome().raw().to_string(),
                variable_name: format!("outcome:{}", world.binding().outcome().outcome().raw()),
                role: QuantityRole::Outcome,
                units: world.binding().outcome().units().name().into(),
                population_id: format!("joint-design-population:{}", self.prepared.dependency),
                regime_id: format!(
                    "compiled_world:{}:{}",
                    self.prepared.target.identity(),
                    world.id().raw()
                ),
                horizon: world.time().horizon_steps().unwrap_or(0),
                functional_id: "mean".into(),
                conditioning: vec![],
                transform_id: "identity".into(),
            })
            .collect::<Vec<_>>();
        Ok(quantities)
    }
    /// Quantity-bound point or posterior means; carries no distributional coverage claim.
    /// # Errors
    /// Changed target quantity bindings or invalid scientific source projection.
    pub fn mean_source(&self) -> Result<ScientificDecisionSource, JointPanelError> {
        let quantities = self.coordinates()?;
        let worlds = contracts::target_worlds(&self.prepared.target)?;
        let payload = antecedent_design::decision_eval::MeanSource {
            coordinates: quantities.clone(),
            means: self.law.estimate().means.clone(),
            estimate: if matches!(self.law.as_ref(), JointDesignLaw::Bayesian(_)) {
                antecedent_design::decision_eval::MeanEstimate::Posterior
            } else {
                antecedent_design::decision_eval::MeanEstimate::Frequentist
            },
            provider_id: self.prepared.proof.recipe.method().into(),
            snapshot_id: self.dependency.to_hex(),
            causal_contract_id: self.prepared.target.identity().to_hex(),
            rng_id: "retained_native_design".into(),
        };
        let basis = worlds
            .iter()
            .zip(&quantities)
            .map(|(w, q)| ScientificProjectionBasis {
                world: w.id(),
                variable: w.binding().outcome().outcome(),
                quantity: q.clone(),
            })
            .collect();
        let outputs = quantities
            .into_iter()
            .enumerate()
            .map(|(i, q)| ScientificProjectionOutput { quantity: q, terms: vec![(i, 1.0)] })
            .collect();
        ScientificDecisionSource::new_projected(
            self.prepared.target.as_ref().clone(),
            self.prepared.identification.as_ref().clone(),
            self.dependency,
            ScientificSourcePayload::Means(payload),
            ScientificSourceProjection { basis, outputs },
        )
        .map(|s| s.retain_admission(self.admission.clone()))
        .map_err(failure)
    }
    /// Build a quantity-bound posterior decision/experiment port from actual aligned mechanism draws.
    /// # Errors
    /// A frequentist moment estimate cannot be consumed as a Bayesian law.
    pub fn posterior_source(&self) -> Result<ScientificDecisionSource, JointPanelError> {
        let JointDesignLaw::Bayesian(posterior) = self.law.as_ref() else {
            return Err(failure("frequentist joint moments are not a Bayesian posterior law"));
        };
        let worlds = contracts::target_worlds(&self.prepared.target)?;
        let quantities = self.coordinates()?;
        let identity = DistributionIdentity::new(
            DistributionMeaningWire::CausalFunctionalPosterior,
            &quantities,
            DrawAlignment::Joint,
            DistributionProvenance {
                source_id: self.prepared.dependency.to_hex(),
                provider_id: self.prepared.proof.recipe.method().into(),
                rng_id: format!("retained:{:?}", self.inference),
                snapshot_id: self.dependency.to_hex(),
                causal_contract_id: self.prepared.target.identity().to_hex(),
            },
        )
        .map_err(failure)?;
        let metadata = DistributionMetadata {
            version: 1,
            identity,
            axes: ["draw".into(), "quantity".into()],
            shape: [posterior.draws.len(), quantities.len()],
            weights: None,
            supported: Some(vec![true; quantities.len()]),
            calibration: DistributionCalibration::Unmeasured,
            trust: DistributionTrust::Unverified,
            legacy_posterior: None,
            legacy_bindings: None,
        };
        let law = DistributionArtifact::new(
            metadata,
            posterior.draws.iter().flatten().copied().collect(),
        )
        .map_err(failure)?;
        let basis = worlds
            .iter()
            .zip(&quantities)
            .map(|(w, q)| ScientificProjectionBasis {
                world: w.id(),
                variable: w.binding().outcome().outcome(),
                quantity: q.clone(),
            })
            .collect();
        let outputs = quantities
            .into_iter()
            .enumerate()
            .map(|(i, q)| ScientificProjectionOutput { quantity: q, terms: vec![(i, 1.0)] })
            .collect();
        ScientificDecisionSource::new_projected(
            self.prepared.target.as_ref().clone(),
            self.prepared.identification.as_ref().clone(),
            self.dependency,
            ScientificSourcePayload::Law(Box::new(law)),
            ScientificSourceProjection { basis, outputs },
        )
        .map(|source| source.retain_admission(self.admission.clone()))
        .map_err(failure)
    }
}

/// Largest portable design artifact under [`DesignArtifactLimits::default`].
pub const MAX_DESIGN_ARTIFACT_BYTES: usize = 16 * 1024 * 1024;

/// Byte bound of a portable design artifact, checked before decoding and after encoding.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DesignArtifactLimits {
    /// Largest artifact body.
    pub max_bytes: usize,
}

impl Default for DesignArtifactLimits {
    fn default() -> Self {
        Self { max_bytes: MAX_DESIGN_ARTIFACT_BYTES }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct DesignReplay {
    version: u16,
    target: antecedent_io::scientific_target_wire::ScientificTargetWire,
    proof: JointDesignProof,
    inference: JointDesignInference,
    law: JointDesignLaw,
}
impl JointDesignResult {
    /// Export the complete original data, typed proof, statistical recipe and aligned law.
    /// Historical replay establishes numerical reproducibility; current causal consumption
    /// additionally requires a caller-retained current prepared design.
    /// # Errors
    /// Encoding failure or a portable artifact exceeding the bounded size.
    pub fn export_artifact(&self) -> Result<Vec<u8>, JointPanelError> {
        self.export_artifact_with_limits(DesignArtifactLimits::default())
    }
    /// [`Self::export_artifact`] under a caller-declared byte limit.
    /// # Errors
    /// As [`Self::export_artifact`], with `limits.max_bytes` as the size bound.
    pub fn export_artifact_with_limits(
        &self,
        limits: DesignArtifactLimits,
    ) -> Result<Vec<u8>, JointPanelError> {
        let wire = DesignReplay {
            version: 1,
            target: antecedent_io::scientific_target_wire::ScientificTargetWire::from_domain(
                &self.prepared.target,
            )
            .map_err(failure)?,
            proof: self.prepared.proof.as_ref().clone(),
            inference: self.inference.as_ref().clone(),
            law: self.law.as_ref().clone(),
        };
        let bytes = antecedent_io::convert::to_cbor(&wire).map_err(failure)?;
        if bytes.len() > limits.max_bytes {
            return Err(failure("joint-design artifact exceeds the declared portable size"));
        }
        Ok(bytes)
    }
}
fn decode_replay(
    bytes: &[u8],
    ctx: &ExecutionContext,
    limits: DesignArtifactLimits,
) -> Result<(DesignReplay, Arc<antecedent_core::execution::AdmissionLease>), JointPanelError> {
    check_cancel(ctx)?;
    if bytes.len() > limits.max_bytes {
        return Err(failure("joint-design artifact exceeds the declared decode size limit"));
    }
    let quote = u64::try_from(bytes.len())
        .ok()
        .and_then(|n| n.checked_mul(128))
        .ok_or_else(|| failure("joint-design decode admission overflow"))?;
    let lease = native::admitted(ctx, quote)?;
    let wire: DesignReplay = antecedent_io::convert::from_cbor(bytes).map_err(failure)?;
    if wire.version != 1 {
        return Err(failure("unsupported joint-design execution artifact version"));
    }
    Ok((wire, lease))
}
/// Independently rerun a frozen design proof and its native likelihood/uncertainty procedure.
/// This returns numerical law only and does not restate historical premises as current evidence.
/// # Errors
/// Malformed artifact, invalid premises/support, changed deterministic output or resource refusal.
pub fn replay_joint_design_artifact(
    bytes: &[u8],
    ctx: &ExecutionContext,
) -> Result<JointDesignLaw, JointPanelError> {
    replay_joint_design_artifact_with_limits(bytes, ctx, DesignArtifactLimits::default())
}

/// [`replay_joint_design_artifact`] under a caller-declared byte limit.
/// # Errors
/// As [`replay_joint_design_artifact`], with `limits.max_bytes` as the size bound.
pub fn replay_joint_design_artifact_with_limits(
    bytes: &[u8],
    ctx: &ExecutionContext,
    limits: DesignArtifactLimits,
) -> Result<JointDesignLaw, JointPanelError> {
    replay_joint_design_execution(bytes, ctx, limits).map(|result| result.law.as_ref().clone())
}

pub(crate) fn replay_joint_design_execution(
    bytes: &[u8],
    ctx: &ExecutionContext,
    limits: DesignArtifactLimits,
) -> Result<JointDesignResult, JointPanelError> {
    let (wire, _lease) = decode_replay(bytes, ctx, limits)?;
    let target = wire.target.to_domain().map_err(failure)?;
    let prepared = PreparedJointDesign::compile(target, wire.proof, ctx)?;
    let result = prepared.execute(wire.inference, ctx)?;
    if result.law.as_ref() != &wire.law {
        return Err(failure(
            "frozen joint-design evidence does not reproduce its retained native law",
        ));
    }
    Ok(result)
}
impl PreparedJointDesign {
    /// Consume historical execution only after matching independently retained current
    /// source data, world bindings, premises and identifying functional.
    /// # Errors
    /// Artifact differs from this current design, or retained numerical evidence fails replay.
    pub fn consume_current(
        &self,
        bytes: &[u8],
        ctx: &ExecutionContext,
    ) -> Result<JointDesignResult, JointPanelError> {
        self.consume_current_with_limits(bytes, ctx, DesignArtifactLimits::default())
    }
    /// [`Self::consume_current`] under a caller-declared byte limit.
    /// # Errors
    /// As [`Self::consume_current`], with `limits.max_bytes` as the size bound.
    pub fn consume_current_with_limits(
        &self,
        bytes: &[u8],
        ctx: &ExecutionContext,
        limits: DesignArtifactLimits,
    ) -> Result<JointDesignResult, JointPanelError> {
        let (wire, _lease) = decode_replay(bytes, ctx, limits)?;
        let target = wire.target.to_domain().map_err(failure)?;
        if identity(&target, &wire.proof)? != self.dependency {
            return Err(failure(
                "joint-design artifact does not match caller-current data, target or design premises",
            ));
        }
        let result = self.execute(wire.inference, ctx)?;
        if result.law.as_ref() != &wire.law {
            return Err(failure("joint-design artifact contains nonreproducible evidence"));
        }
        Ok(result)
    }
}

#[cfg(test)]
pub(crate) use native::{gaussian_longitudinal_chain_fixture, native_experiment_fixtures};
