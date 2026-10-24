//! Reference decomposition from an authentic compiled joint scientific source.
// SPDX-License-Identifier: MIT OR Apache-2.0
use super::{ChangeComponent, JointPanelError, PanelDecomposition};
use antecedent_core::{
    ExecutionContext, Intervention, ScientificWorldId, SemanticDigest, SequencedIntervention,
    StructuralAction, TemporalPolicy, VariableId, WorldAction,
};
use antecedent_design::scientific_source::{
    ScientificConsumptionScope, ScientificDecisionSource, ScientificProjectionBasis,
    ScientificProjectionOutput, ScientificSourcePayload, ScientificSourceProjection,
};
use antecedent_io_core::distribution_artifact::DrawAlignment;

/// Explicit disjoint component bindings in the scientific intervention language.
/// Population is the world's population binding, rather than a hidden mechanism input.
#[derive(Clone, Debug)]
pub struct ReferenceComponentBindings {
    /// Schedules, clamps and initial-state actions assigned to exposure.
    pub exposure: Vec<VariableId>,
    /// Saturation parameter nodes or mechanism replacement targets.
    pub saturation: Vec<VariableId>,
    /// All other changing response/mechanism nodes.
    pub mechanism: Vec<VariableId>,
    /// Exact source worlds by exposure/saturation/population/mechanism bit mask.
    pub hybrids: [ScientificWorldId; 16],
    /// Caller-requested component replacement ordering.
    pub order: [ChangeComponent; 4],
}
/// Scientific lineage and complete aligned reference-world decomposition.
#[derive(Clone, Debug)]
pub struct ScientificReferenceDecomposition {
    /// Original full-procedure resampling failures, retaining every requested slot.
    pub procedure_failures: Vec<Option<String>>,
    /// Complete numerical decomposition, with original joint uncertainty semantics.
    pub decomposition: PanelDecomposition,
    /// Analytically propagated native joint covariance, when supplied by the method owner.
    pub covariance: Option<super::engine::decomposition::ReferenceDecompositionCovariance>,
    /// Actual source retained for downstream authorization and artifact replay.
    pub source: ScientificDecisionSource,
    /// Exact declaration, including all hybrid identities and separate component bindings.
    pub bindings: ReferenceComponentBindings,
    /// Scope actually checked before reading the numerical source.
    pub scope: ScientificConsumptionScope,
    /// Exact compiled source question, including sparse projection lineage if present.
    pub question_identity: SemanticDigest,
}
#[derive(Clone, Debug, PartialEq)]
enum Actions {
    Empty,
    Static(Vec<Intervention>),
    Scheduled(TemporalPolicy, Vec<SequencedIntervention>),
    Structural(Vec<StructuralAction>),
}
fn invalid(reason: impl Into<String>) -> JointPanelError {
    JointPanelError::compile(reason.into())
}
fn block(
    variable: VariableId,
    bindings: &ReferenceComponentBindings,
) -> Result<usize, JointPanelError> {
    [bindings.exposure.as_slice(), bindings.saturation.as_slice(), bindings.mechanism.as_slice()]
        .iter()
        .position(|variables| variables.contains(&variable))
        .ok_or_else(|| invalid("every acted-on variable requires an explicit component binding"))
}
fn split(
    action: &WorldAction,
    bindings: &ReferenceComponentBindings,
) -> Result<[Actions; 3], JointPanelError> {
    let classify = |intervention: &Intervention| -> Result<usize, JointPanelError> {
        let variables = intervention.target_variables();
        let first = variables
            .first()
            .copied()
            .ok_or_else(|| invalid("component action has no target variable"))?;
        let component = block(first, bindings)?;
        if variables.iter().any(|variable| block(*variable, bindings).ok() != Some(component)) {
            return Err(invalid("one intervention crosses component bindings"));
        }
        Ok(component)
    };
    if let Some(interventions) = action.at_decision_interventions() {
        let mut groups: [Vec<Intervention>; 3] = std::array::from_fn(|_| Vec::new());
        for intervention in interventions {
            groups[classify(intervention)?].push(intervention.clone());
        }
        return Ok(groups.map(|values| {
            if values.is_empty() {
                Actions::Empty
            } else {
                Actions::Static(values)
            }
        }));
    }
    if let Some((intervention, policy)) = action.schedule_parts() {
        let mut groups: [Vec<SequencedIntervention>; 3] = std::array::from_fn(|_| Vec::new());
        if let Intervention::Sequence(sequence) = intervention {
            for step in sequence.steps.iter() {
                groups[classify(&step.intervention)?].push(step.clone());
            }
        } else {
            groups[classify(intervention)?]
                .push(SequencedIntervention::new(intervention.clone(), policy.clone()));
        }
        return Ok(groups.map(|steps| {
            if steps.is_empty() {
                Actions::Empty
            } else {
                Actions::Scheduled(policy.clone(), steps)
            }
        }));
    }
    if let Some(actions) = action.structural_actions() {
        let mut groups: [Vec<StructuralAction>; 3] = std::array::from_fn(|_| Vec::new());
        for action in actions {
            groups[block(action.variable(), bindings)?].push(action.clone());
        }
        return Ok(groups.map(|values| {
            if values.is_empty() {
                Actions::Empty
            } else {
                Actions::Structural(values)
            }
        }));
    }
    Err(invalid("unrecognized reference-world action"))
}
/// Decompose the sixteen genuinely compiled hybrid worlds of any compatible method owner.
///
/// The source compiler must earn each world's identification and transport claim.
/// This adapter checks the binary hybrid construction itself, exact population binding,
/// clock/history, outcome and unit, disjoint intervention components, and aligned law.
/// It never creates a proof from caller assertions or independently paired marginals.
/// A law's point is its coordinate mean; means-only sources retain their actual point.
///
/// # Errors
/// Unsupported scientific scope, incorrect/missing hybrid, component overlap,
/// independent marginals, numerical/resource/cancellation failure.
#[expect(
    clippy::too_many_lines,
    reason = "validate the complete compiled hybrid basis before consuming its aligned law"
)]
pub fn decompose_scientific_reference(
    source: &ScientificDecisionSource,
    bindings: &ReferenceComponentBindings,
    scope: ScientificConsumptionScope,
    ctx: &ExecutionContext,
) -> Result<ScientificReferenceDecomposition, JointPanelError> {
    source.authorize(scope).map_err(JointPanelError::compile)?;
    let mut seen = std::collections::BTreeSet::new();
    for variable in bindings.exposure.iter().chain(&bindings.saturation).chain(&bindings.mechanism)
    {
        if !seen.insert(*variable) {
            return Err(invalid("component variable bindings must be disjoint and duplicate-free"));
        }
    }
    let worlds = super::contracts::target_worlds(source.target())?;
    let world = |id| {
        worlds
            .iter()
            .find(|world| world.id() == id)
            .ok_or_else(|| invalid("reference hybrid names an absent compiled world"))
    };
    let before = world(bindings.hybrids[0])?;
    let after = world(bindings.hybrids[15])?;
    let before_actions = split(before.action(), bindings)?;
    let after_actions = split(after.action(), bindings)?;
    for (mask, id) in bindings.hybrids.iter().enumerate() {
        if ctx.cancellation.is_cancelled() {
            return Err(super::engine::PanelError::Cancelled.into());
        }
        let hybrid = world(*id)?;
        let actual = split(hybrid.action(), bindings)?;
        for (component, bit) in [1, 2, 8].into_iter().enumerate() {
            let expected = if mask & bit == 0 {
                &before_actions[component]
            } else {
                &after_actions[component]
            };
            if &actual[component] != expected {
                return Err(invalid(format!("hybrid {mask} mixes or omits component {component}")));
            }
        }
        let population = if mask & 4 == 0 { before } else { after };
        let exposure = if mask & 1 == 0 { before } else { after };
        if hybrid.binding().population() != population.binding().population()
            || hybrid.binding().conditioning() != population.binding().conditioning()
            || hybrid.binding().outcome() != before.binding().outcome()
            || hybrid.binding().unit() != before.binding().unit()
            || hybrid.time() != exposure.time()
        {
            return Err(invalid(format!(
                "hybrid {mask} changes an undeclared population, outcome, unit or exposure clock/history"
            )));
        }
    }
    let expected = bindings
        .hybrids
        .map(|id| format!("compiled_world:{}:{}", source.target().identity(), id.raw()));
    let slots = match source.as_numerical_source() {
        ScientificSourcePayload::Means(_) => 0,
        ScientificSourcePayload::Law(law) => law.draws().len() / law.quantities().len(),
    };
    let scratch_bytes = slots
        .checked_mul(std::mem::size_of::<Option<[f64; 16]>>() + std::mem::size_of::<f64>())
        .and_then(|n| n.checked_add(16 * 128))
        .ok_or_else(|| invalid("reference law scratch overflow"))?;
    let _scratch = ctx
        .acquire_admission(0, scratch_bytes as u64)
        .map_err(|error| super::engine::PanelError::Resource(error.to_string()))?;
    let (point, draws, weights) = match source.as_numerical_source() {
        ScientificSourcePayload::Means(means) => {
            if means.coordinates.len() != means.means.len()
                || means.means.iter().any(|value| !value.is_finite())
            {
                return Err(invalid("malformed scientific mean source"));
            }
            let mut point = [0.0; 16];
            for j in 0..16 {
                let index = means
                    .coordinates
                    .iter()
                    .position(|quantity| {
                        quantity.regime_id == expected[j] && quantity.functional_id == "mean"
                    })
                    .ok_or_else(|| {
                        invalid("source is missing an unprojected hybrid mean coordinate")
                    })?;
                point[j] = means.means[index];
            }
            (point, Vec::new(), None)
        }
        ScientificSourcePayload::Law(law) => {
            if law.metadata().identity.alignment != DrawAlignment::Joint {
                return Err(invalid(
                    "reference decomposition requires genuinely aligned joint worlds",
                ));
            }
            let mut indices = [0; 16];
            for j in 0..16 {
                indices[j] = law
                    .quantities()
                    .iter()
                    .position(|quantity| {
                        quantity.regime_id == expected[j] && quantity.functional_id == "mean"
                    })
                    .ok_or_else(|| invalid("joint law lacks a hybrid mean coordinate"))?;
            }
            if law
                .metadata()
                .supported
                .as_ref()
                .is_some_and(|supported| indices.iter().any(|index| !supported[*index]))
            {
                return Err(invalid("a required hybrid law coordinate is unsupported"));
            }
            let width = law.quantities().len();
            let mut point = [0.0; 16];
            let weights = law.metadata().weights.clone();
            let total_weight = weights
                .as_ref()
                .map_or(law.metadata().shape[0] as f64, |weights| weights.iter().sum());
            let draws = law
                .draws()
                .chunks_exact(width)
                .enumerate()
                .map(|(index, row)| {
                    let values = std::array::from_fn(|j| row[indices[j]]);
                    for j in 0..16 {
                        point[j] += values[j]
                            * (weights.as_ref().map_or(1.0, |weights| weights[index])
                                / total_weight);
                    }
                    Some(values)
                })
                .collect::<Vec<_>>();
            (point, draws, weights)
        }
    };
    let decomposition = super::engine::decomposition::decompose_weighted_hybrid_law(
        point,
        &draws,
        weights.as_deref(),
        bindings.order,
        ctx,
    )?;
    Ok(ScientificReferenceDecomposition {
        procedure_failures: Vec::new(),
        decomposition,
        covariance: None,
        source: source.clone(),
        bindings: bindings.clone(),
        scope,
        question_identity: source.question_identity(),
    })
}

impl super::JointPanelResult {
    /// Apply explicit component bindings to the actual compiled terminal world means.
    /// Source data, identification, mechanism, joint law and uncertainty semantics remain
    /// those of this result. Arbitrary scenario weights or fitted parameter surgery are
    /// outside this world basis and must be compiled by their own method owner.
    /// # Errors
    /// Missing hybrid worlds, unsupported scope, malformed bindings or numerical resources.
    pub fn decompose_compiled_components(
        &self,
        bindings: &ReferenceComponentBindings,
        scope: ScientificConsumptionScope,
        with_uncertainty: bool,
        ctx: &ExecutionContext,
    ) -> Result<ScientificReferenceDecomposition, JointPanelError> {
        if !bindings.saturation.is_empty()
            || !bindings.mechanism.is_empty()
            || bindings
                .exposure
                .iter()
                .any(|variable| !self.request().columns.channels.contains(variable))
        {
            return Err(invalid(
                "panel program action variables are exposure channels; saturation/mechanism changes require explicit parameter or structural-action worlds from their owning compiler",
            ));
        }
        let mut means = self.numerical_project_mean_source(
            &super::super::JointPanelResponseFunctional::TerminalMean,
        )?;
        let target_worlds = super::contracts::target_worlds(&self.request().target)?;
        let mut basis = Vec::with_capacity(self.request().worlds.len());
        for (quantity, world) in means.coordinates.iter_mut().zip(&self.request().worlds) {
            let target = target_worlds
                .iter()
                .find(|target| target.id() == world.world)
                .ok_or_else(|| invalid("missing compiled terminal world"))?;
            quantity.regime_id = format!(
                "compiled_world:{}:{}",
                self.request().target.identity(),
                world.world.raw()
            );
            quantity.functional_id = "mean".into();
            quantity.horizon = target.time().horizon_steps().unwrap_or(0);
            basis.push(ScientificProjectionBasis {
                world: world.world,
                variable: target.binding().outcome().outcome(),
                quantity: quantity.clone(),
            });
        }
        let payload = if with_uncertainty {
            let law = self.numerical_project_law_source(
                &super::super::JointPanelResponseFunctional::TerminalMean,
            )?;
            let mut metadata = law.metadata().clone();
            metadata.identity.quantities = means
                .coordinates
                .iter()
                .map(antecedent_io_core::quantity_wire::ScientificQuantityWire::from)
                .collect();
            ScientificSourcePayload::Law(Box::new(
                antecedent_io_core::distribution_artifact::DistributionArtifact::new(
                    metadata,
                    law.draws().to_vec(),
                )
                .map_err(JointPanelError::compile)?,
            ))
        } else {
            ScientificSourcePayload::Means(means.clone())
        };
        let projection = ScientificSourceProjection {
            basis,
            outputs: means
                .coordinates
                .into_iter()
                .enumerate()
                .map(|(index, quantity)| ScientificProjectionOutput {
                    quantity,
                    terms: vec![(index, 1.0)],
                })
                .collect(),
        };
        let source = ScientificDecisionSource::new_projected(
            self.request().target.clone(),
            self.identification().clone(),
            self.dependency_identity(),
            payload,
            projection,
        )
        .map_err(JointPanelError::compile)?;
        decompose_scientific_reference(&source, bindings, scope, ctx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::joint_panel::designs::{
        JointDesignProof, JointDesignWorldBinding, PreparedJointDesign,
    };
    use antecedent_core::{
        Conditioning, GraphFamily, InferenceSelection, InterventionSequence, InterventionWorld,
        ModelAssumptions, ObservationUnit, OutcomeQuantity, QuantityUnits, ScientificTarget,
        TargetForm, TargetPopulation, TimeFrame, Value, WorldBinding, WorldCollection,
    };
    use antecedent_estimate::joint_design::compiled::{JointDesignInference, JointDesignRecipe};
    use antecedent_estimate::joint_design::randomized::{
        JointRandomizedDesign, JointRandomizedUnit, RandomizedGaussianPrior,
    };

    #[test]
    fn continuous_history_bootstrap_and_posterior_reference_preserve_the_actual_law() {
        let ctx = ExecutionContext::for_tests(559);
        let (recipe, bayesian, _) =
            crate::analysis::joint_panel::designs::gaussian_longitudinal_chain_fixture();
        let JointDesignRecipe::GaussianLongitudinal { design, .. } = &recipe else {
            panic!("continuous history owner")
        };
        let worlds = design
            .worlds
            .iter()
            .enumerate()
            .map(|(index, program)| {
                let steps: Vec<_> = program
                    .treatments
                    .iter()
                    .enumerate()
                    .map(|(time, values)| {
                        SequencedIntervention::new(
                            Intervention::set(VariableId::from_raw(1), Value::f64(values[0])),
                            TemporalPolicy::pulse(time as i32),
                        )
                    })
                    .collect();
                InterventionWorld::new(
                    ScientificWorldId::from_raw(index as u32),
                    WorldAction::schedule(
                        Intervention::Sequence(InterventionSequence::new(steps)),
                        TemporalPolicy::sustained(0, 1),
                    )
                    .unwrap(),
                    index == 0,
                    WorldBinding::new(
                        TargetPopulation::AllObserved,
                        ObservationUnit::new("independent episode").unwrap(),
                        OutcomeQuantity::mean(
                            VariableId::from_raw(0),
                            QuantityUnits::new("response").unwrap(),
                        )
                        .unwrap(),
                        Conditioning::unconditional(),
                    )
                    .unwrap(),
                    TimeFrame::temporal(
                        2,
                        antecedent_core::HistoryWindow::max_lag(0),
                        antecedent_core::InitialLaw::named("observed baseline history").unwrap(),
                    )
                    .unwrap(),
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let source_binding =
            crate::analysis::joint_panel::designs::JointDesignSourceBinding::from_world(
                &worlds[0],
                design.population.clone(),
            )
            .unwrap();
        let bindings = ReferenceComponentBindings {
            exposure: vec![VariableId::from_raw(1)],
            saturation: vec![],
            mechanism: vec![],
            hybrids: std::array::from_fn(|mask| {
                ScientificWorldId::from_raw(if mask & 1 == 0 { 0 } else { 3 })
            }),
            order: [
                ChangeComponent::Exposure,
                ChangeComponent::Saturation,
                ChangeComponent::Population,
                ChangeComponent::Mechanism,
            ],
        };
        let proof = JointDesignProof {
            version: 1,
            source_binding,
            longitudinal_arms: vec![],
            recipe,
            worlds: worlds
                .iter()
                .enumerate()
                .map(|(index, world)| JointDesignWorldBinding {
                    key: index.to_string(),
                    world: index as u32,
                    target_identity:
                        crate::analysis::joint_panel::designs::joint_design_world_identity(world)
                            .to_hex(),
                })
                .collect(),
        };
        let target = ScientificTarget::new(
            TargetForm::Worlds(WorldCollection::new(worlds).unwrap()),
            ModelAssumptions::new(GraphFamily::Unspecified, Default::default()),
            InferenceSelection::Bayesian,
        );
        let prepared = PreparedJointDesign::compile(target, proof, &ctx).unwrap();
        for inference in [
            JointDesignInference::GaussianLongitudinalBootstrap { replicates: 32, seed: 559 },
            bayesian,
        ] {
            let result = prepared.execute(inference, &ctx).unwrap();
            let decomposition = result
                .decompose_reference_components(
                    &bindings,
                    ScientificConsumptionScope::IdentifiedCausal,
                    &ctx,
                )
                .unwrap();
            assert!((decomposition.decomposition.point.total - 16.6).abs() < 0.5);
            assert!(decomposition.decomposition.draws.iter().flatten().all(|draw| (draw.shapley
                [0]
                - draw.total)
                .abs()
                < 1e-10
                && draw.shapley[1..].iter().all(|value| value.abs() < 1e-10)));
            if let Some(rows) = result.law().frequentist_realizations() {
                assert_eq!(decomposition.decomposition.draws.len(), rows.len());
                assert_eq!(
                    decomposition.procedure_failures,
                    result.law().frequentist_failures().unwrap()
                );
                assert_eq!(
                    decomposition.decomposition.point.hybrids[1],
                    result.law().estimate().means[3]
                );
                for (actual, native) in decomposition.decomposition.draws.iter().zip(rows) {
                    assert_eq!(actual.is_some(), native.is_some());
                    if let (Some(actual), Some(native)) = (actual, native) {
                        assert!((actual.total - (native[3] - native[0])).abs() < 1e-10);
                    }
                }
            } else {
                assert_eq!(decomposition.decomposition.draws.len(), 512);
            }
            let artifact = result
                .export_reference_decomposition(
                    &bindings,
                    ScientificConsumptionScope::IdentifiedCausal,
                    &ctx,
                )
                .unwrap();
            let historical = replay_reference_decomposition(&artifact, &ctx).unwrap();
            assert_eq!(historical.decomposition.draws, decomposition.decomposition.draws);
            assert_eq!(historical.procedure_failures, decomposition.procedure_failures);
        }
    }

    fn compiled_parameter_experiment(
        temporal: bool,
    ) -> (super::super::super::designs::JointDesignResult, ReferenceComponentBindings) {
        compiled_parameter_experiment_scenario(temporal, None, 8192)
    }
    fn compiled_parameter_experiment_scenario(
        temporal: bool,
        study_seed: Option<u64>,
        posterior_draws: usize,
    ) -> (super::super::super::designs::JointDesignResult, ReferenceComponentBindings) {
        compiled_parameter_population_scenario(temporal, study_seed, posterior_draws, false)
    }
    fn compiled_parameter_population_scenario(
        temporal: bool,
        study_seed: Option<u64>,
        posterior_draws: usize,
        population_strata: bool,
    ) -> (super::super::super::designs::JointDesignResult, ReferenceComponentBindings) {
        let arms = if population_strata { 16 } else { 8 };
        let variables = if population_strata { 4 } else { 3 };
        let ctx = ExecutionContext::for_tests(311);
        let levels = |mask: usize| {
            [
                if mask & 1 == 0 { 1.0 } else { 4.0 },
                if mask & 2 == 0 { 1.0 } else { 2.0 },
                if mask & 4 == 0 { 1.0 } else { 3.0 },
                if population_strata && mask & 8 != 0 { 1.0 } else { 0.0 },
            ]
        };
        let worlds = (0..arms)
            .map(|mask| {
                let values = levels(mask);
                let interventions = (0..variables)
                    .map(|j| {
                        Intervention::set(VariableId::from_raw(j as u32), Value::f64(values[j]))
                    })
                    .collect::<Vec<_>>();
                let action = if temporal {
                    WorldAction::schedule(
                        Intervention::Sequence(InterventionSequence::new(
                            (0..2)
                                .flat_map(|offset| {
                                    interventions.iter().cloned().map(move |intervention| {
                                        SequencedIntervention::new(
                                            intervention,
                                            TemporalPolicy::pulse(offset),
                                        )
                                    })
                                })
                                .collect::<Vec<_>>(),
                        )),
                        TemporalPolicy::sustained(0, 1),
                    )
                    .unwrap()
                } else {
                    WorldAction::at_decision(interventions).unwrap()
                };
                let time = if temporal {
                    TimeFrame::temporal(
                        2,
                        antecedent_core::HistoryWindow::max_lag(0),
                        antecedent_core::InitialLaw::named("shared-zero-history").unwrap(),
                    )
                    .unwrap()
                } else {
                    TimeFrame::static_decision()
                };
                InterventionWorld::new(
                    ScientificWorldId::from_raw(mask as u32),
                    action,
                    mask == 0,
                    WorldBinding::new(
                        TargetPopulation::AllObserved,
                        ObservationUnit::new("independent participant").unwrap(),
                        OutcomeQuantity::mean(
                            VariableId::from_raw(variables as u32),
                            QuantityUnits::new("response").unwrap(),
                        )
                        .unwrap(),
                        Conditioning::unconditional(),
                    )
                    .unwrap(),
                    time,
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let target = ScientificTarget::new(
            TargetForm::Worlds(WorldCollection::new(worlds).unwrap()),
            ModelAssumptions::new(GraphFamily::Unspecified, Default::default()),
            InferenceSelection::Bayesian,
        );
        let design=JointRandomizedDesign { worlds:(0..arms).map(|arm|format!("parameter-arm-{arm}")).collect(),sampling_unit:"independent participant".into(),population:"randomized observed participant population".into(),consistency:"physical interventions jointly set dose A, saturation-control node S and mechanism-control node M; final response measured under each full program".into(),between_unit_isolation:"participants have isolated responses".into() };
        let units = (0..arms)
            .flat_map(|arm| {
                (0..4).map(move |repeat| {
                    let values = levels(arm);
                    JointRandomizedUnit {
                        id: format!("{arm}/{repeat}"),
                        arm,
                        probabilities: vec![1.0 / arms as f64; arms],
                        outcome: (1.0 + values[3]) * values[2] * values[0]
                            / (values[0] + values[1])
                            + if repeat % 2 == 0 { -0.1 } else { 0.1 },
                        population_weight: 1.0,
                    }
                })
            })
            .collect::<Vec<_>>();
        let units = if let Some(seed) = study_seed {
            let mut rng = antecedent_core::CausalRng::from_seed(seed);
            (0..512)
                .map(|unit| {
                    let arm = (rng.next_u64() % arms as u64) as usize;
                    let values = levels(arm);
                    let noise = (-2.0 * rng.next_f64().max(f64::MIN_POSITIVE).ln()).sqrt()
                        * (std::f64::consts::TAU * rng.next_f64()).cos();
                    JointRandomizedUnit {
                        id: format!("unit-{unit}"),
                        arm,
                        probabilities: vec![1.0 / arms as f64; arms],
                        outcome: (1.0 + values[3]) * values[2] * values[0]
                            / (values[0] + values[1])
                            + 0.2 * noise,
                        population_weight: 1.0,
                    }
                })
                .collect::<Vec<_>>()
        } else {
            units
        };
        let proof = JointDesignProof {
            version: 1,
            source_binding:
                crate::analysis::joint_panel::designs::JointDesignSourceBinding::from_world(
                    &super::super::contracts::target_worlds(&target).unwrap()[0],
                    "randomized observed participant population".into(),
                )
                .unwrap(),
            longitudinal_arms: vec![],
            recipe: JointDesignRecipe::Randomized { design, units: units.clone() },
            worlds: (0..arms)
                .map(|arm| JointDesignWorldBinding {
                    key: format!("parameter-arm-{arm}"),
                    world: arm as u32,
                    target_identity:
                        crate::analysis::joint_panel::designs::joint_design_world_identity(
                            &super::super::contracts::target_worlds(&target).unwrap()[arm as usize],
                        )
                        .to_hex(),
                })
                .collect(),
        };
        let prepared = PreparedJointDesign::compile(target, proof, &ctx).unwrap();
        let result = prepared
            .execute(
                JointDesignInference::RandomizedGaussian {
                    prior: RandomizedGaussianPrior {
                        mean: vec![0.0; arms],
                        variance: vec![1e6; arms],
                        outcome_variance: vec![0.04; units.len()],
                        draws: posterior_draws,
                        seed: study_seed.unwrap_or(732) ^ 0x25A931,
                    },
                },
                &ctx,
            )
            .unwrap();
        let bindings = ReferenceComponentBindings {
            exposure: vec![VariableId::from_raw(0)],
            saturation: vec![VariableId::from_raw(1)],
            mechanism: vec![VariableId::from_raw(2)],
            hybrids: std::array::from_fn(|mask| {
                ScientificWorldId::from_raw(((mask & 3) | ((mask & 8) >> 1)) as u32)
            }),
            order: [
                ChangeComponent::Exposure,
                ChangeComponent::Saturation,
                ChangeComponent::Population,
                ChangeComponent::Mechanism,
            ],
        };
        (result, bindings)
    }
    #[test]
    fn actual_parameter_factorial_reference_decomposes_static_and_temporal_posterior_sources() {
        for temporal in [false, true] {
            let (result, bindings) = compiled_parameter_experiment(temporal);
            let frequentist = result
                .prepared()
                .execute(JointDesignInference::Frequentist, &ExecutionContext::for_tests(313))
                .unwrap();
            let frequentist_decomposition = frequentist
                .decompose_reference_components(
                    &bindings,
                    ScientificConsumptionScope::IdentifiedCausal,
                    &ExecutionContext::for_tests(313),
                )
                .unwrap();
            let frequentist_covariance = frequentist_decomposition.covariance.as_ref().unwrap();
            assert!(
                (frequentist_covariance.shapley.iter().flatten().sum::<f64>()
                    - frequentist_covariance.total)
                    .abs()
                    < 1e-10
            );
            assert!(frequentist_covariance.shapley[2].iter().all(|value| value.abs() < 1e-12));
            let source = result.posterior_source().unwrap();
            let decomposition = decompose_scientific_reference(
                &source,
                &bindings,
                ScientificConsumptionScope::IdentifiedCausal,
                &ExecutionContext::for_tests(312),
            )
            .unwrap();
            assert!(decomposition.decomposition.draws.len() == 8192);
            let truth: [f64; 16] = std::array::from_fn(|mask| {
                let a = if mask & 1 == 0 { 1.0 } else { 4.0 };
                let s = if mask & 2 == 0 { 1.0 } else { 2.0 };
                let m = if mask & 8 == 0 { 1.0 } else { 3.0 };
                m * a / (a + s)
            });
            for component in 0..4 {
                let independent = (0_usize..16)
                    .filter(|mask| mask & (1 << component) == 0)
                    .map(|mask| {
                        let weight = match mask.count_ones() {
                            0 | 3 => 0.25,
                            1 | 2 => 1.0 / 12.0,
                            _ => unreachable!(),
                        };
                        weight * (truth[mask | (1 << component)] - truth[mask])
                    })
                    .sum::<f64>();
                assert!(
                    (frequentist_decomposition.decomposition.point.shapley[component]
                        - independent)
                        .abs()
                        < 1e-12
                );
                assert!(
                    (decomposition.decomposition.point.shapley[component] - independent).abs()
                        < 0.01,
                    "component={component}, temporal={temporal}"
                );
            }
            for draw in decomposition.decomposition.draws.iter().flatten() {
                assert!((draw.shapley.iter().sum::<f64>() - draw.total).abs() < 1e-10);
                assert!(draw.shapley[2].abs() < 1e-12);
            }
            assert!(
                decomposition.decomposition.point.order_range[0][1]
                    - decomposition.decomposition.point.order_range[0][0]
                    > 0.1
            );
            let mut invalid = bindings.clone();
            invalid.hybrids[3] = invalid.hybrids[0];
            assert!(decompose_scientific_reference(
                &source,
                &invalid,
                ScientificConsumptionScope::IdentifiedCausal,
                &ExecutionContext::for_tests(312)
            )
            .is_err());
            let reference_artifact = result
                .export_reference_decomposition(
                    &bindings,
                    ScientificConsumptionScope::IdentifiedCausal,
                    &ExecutionContext::for_tests(314),
                )
                .unwrap();
            let historical = replay_reference_decomposition(
                &reference_artifact,
                &ExecutionContext::for_tests(314),
            )
            .unwrap();
            let current_reference = result
                .consume_reference_decomposition_current(
                    &reference_artifact,
                    &bindings,
                    ScientificConsumptionScope::IdentifiedCausal,
                    &ExecutionContext::for_tests(314),
                )
                .unwrap();
            assert_eq!(historical.decomposition.point, current_reference.decomposition.point);
            assert_eq!(historical.decomposition.draws, current_reference.decomposition.draws);
            assert_eq!(historical.covariance, current_reference.covariance);
            let mut changed_order = bindings.clone();
            changed_order.order.swap(0, 1);
            assert!(result
                .consume_reference_decomposition_current(
                    &reference_artifact,
                    &changed_order,
                    ScientificConsumptionScope::IdentifiedCausal,
                    &ExecutionContext::for_tests(314)
                )
                .is_err());
            assert!(result
                .consume_reference_decomposition_current(
                    &reference_artifact,
                    &bindings,
                    ScientificConsumptionScope::ModelConditional,
                    &ExecutionContext::for_tests(314)
                )
                .is_err());
            let bytes = result.export_artifact().unwrap();
            let replay = crate::analysis::joint_panel::designs::replay_joint_design_artifact(
                &bytes,
                &ExecutionContext::for_tests(312),
            )
            .unwrap();
            assert_eq!(&replay, result.law());
            let current = result
                .prepared()
                .consume_current(&bytes, &ExecutionContext::for_tests(312))
                .unwrap();
            let replay_source = current.posterior_source().unwrap();
            let repeated = decompose_scientific_reference(
                &replay_source,
                &bindings,
                ScientificConsumptionScope::IdentifiedCausal,
                &ExecutionContext::for_tests(312),
            )
            .unwrap();
            assert_eq!(decomposition.decomposition.point, repeated.decomposition.point);
            assert_eq!(decomposition.decomposition.draws, repeated.decomposition.draws);
        }
    }
    #[test]
    fn actual_population_transport_decomposition_is_positive_in_both_modes_and_time_forms() {
        use antecedent_graph::{Admg, DenseNodeId, SelectionDiagram};
        for temporal in [false, true] {
            let (posterior, components) =
                compiled_parameter_population_scenario(temporal, None, 8192, true);
            let mut graph = Admg::with_variables(5);
            for variable in 0..4 {
                graph
                    .insert_directed(DenseNodeId::from_raw(variable), DenseNodeId::from_raw(4))
                    .unwrap();
            }
            let diagram =
                SelectionDiagram::try_new(graph.clone(), [VariableId::from_raw(3)]).unwrap();
            let bindings = PopulationReferenceBindings {
                baseline: VariableId::from_raw(3),
                strata: vec![0.0, 1.0],
                stratum_worlds: vec![
                    components.hybrids,
                    components.hybrids.map(|id| ScientificWorldId::from_raw(id.raw() + 8)),
                ],
                before_weights: vec![0.8, 0.2],
                after_weights: vec![0.2, 0.8],
                components: components.clone(),
            };
            if !temporal {
                if let Some(path) = std::env::var_os("ANTECEDENT_POPULATION_FIXTURE_OUTPUT") {
                    let source = posterior.mean_source().unwrap();
                    let target =
                        antecedent_io::scientific_target_wire::ScientificTargetWire::from_domain(
                            source.target(),
                        )
                        .unwrap();
                    let declaration = serde_json::json!({"baseline":3,"strata":[0.0,1.0],"stratum_worlds":bindings.stratum_worlds.iter().map(|worlds|worlds.map(|id|id.raw())).collect::<Vec<_>>(),"before_weights":bindings.before_weights,"after_weights":bindings.after_weights,"exposure":[0],"saturation":[1],"mechanism":[2],"order":["exposure","saturation","population","mechanism"],"diagram":{"node_count":5,"directed":[[0,4],[1,4],[2,4],[3,4]],"bidirected":[],"selection_targets":[3]},"require_identified":true});
                    let fixture = serde_json::json!({"schema":1,"target":target,"proof":posterior.prepared().proof(),"population_reference":declaration,"bayesian":{"inference":"randomized_gaussian","prior":{"mean":vec![0.0;16],"variance":vec![1e6;16],"outcome_variance":vec![0.04;64],"draws":512,"seed":986311}},"known_truth":{"response":"(1+L)*M*A/(A+S)","before_population":[0.8,0.2],"after_population":[0.2,0.8]}});
                    std::fs::write(path, serde_json::to_vec_pretty(&fixture).unwrap()).unwrap();
                }
            }
            let frequentist = posterior
                .prepared()
                .execute(JointDesignInference::Frequentist, &ExecutionContext::for_tests(319))
                .unwrap();
            for result in [&frequentist, &posterior] {
                let decomposition = result
                    .decompose_population_reference(
                        &bindings,
                        &diagram,
                        ScientificConsumptionScope::IdentifiedCausal,
                        &ExecutionContext::for_tests(319),
                    )
                    .unwrap();
                let truth: [f64; 16] = std::array::from_fn(|mask| {
                    let a = if mask & 1 == 0 { 1.0 } else { 4.0 };
                    let s = if mask & 2 == 0 { 1.0 } else { 2.0 };
                    let m = if mask & 8 == 0 { 1.0 } else { 3.0 };
                    let population = if mask & 4 == 0 { 1.2 } else { 1.8 };
                    population * m * a / (a + s)
                });
                for component in 0..4 {
                    let independent = (0_usize..16)
                        .filter(|mask| mask & (1 << component) == 0)
                        .map(|mask| {
                            let weight = match mask.count_ones() {
                                0 | 3 => 0.25,
                                1 | 2 => 1.0 / 12.0,
                                _ => unreachable!(),
                            };
                            weight * (truth[mask | (1 << component)] - truth[mask])
                        })
                        .sum::<f64>();
                    let tolerance = if matches!(
                        result.law(),
                        antecedent_estimate::joint_design::compiled::JointDesignLaw::Frequentist(_)
                    ) {
                        1e-12
                    } else {
                        0.015
                    };
                    assert!(
                        (decomposition.decomposition.point.shapley[component] - independent).abs()
                            < tolerance,
                        "temporal={temporal}, component={component}"
                    );
                }
                assert!(decomposition.decomposition.point.shapley[2] > 0.5);
                let covariance = decomposition.covariance.as_ref().unwrap();
                assert!(
                    (covariance.shapley.iter().flatten().sum::<f64>() - covariance.total).abs()
                        < 1e-10
                );
                for draw in decomposition.decomposition.draws.iter().flatten() {
                    assert!((draw.shapley.iter().sum::<f64>() - draw.total).abs() < 1e-10);
                }
                let bytes = result
                    .export_population_reference(
                        &bindings,
                        &diagram,
                        ScientificConsumptionScope::IdentifiedCausal,
                        &ExecutionContext::for_tests(319),
                    )
                    .unwrap();
                let archived =
                    replay_population_reference(&bytes, &ExecutionContext::for_tests(319)).unwrap();
                let current = result
                    .consume_population_reference_current(
                        &bytes,
                        &bindings,
                        &diagram,
                        ScientificConsumptionScope::IdentifiedCausal,
                        &ExecutionContext::for_tests(319),
                    )
                    .unwrap();
                assert_eq!(decomposition.decomposition.point, archived.decomposition.point);
                assert_eq!(decomposition.decomposition.draws, current.decomposition.draws);
                assert_eq!(decomposition.covariance, current.covariance);
                let mut changed = bindings.clone();
                changed.after_weights = vec![0.4, 0.6];
                assert!(result
                    .consume_population_reference_current(
                        &bytes,
                        &changed,
                        &diagram,
                        ScientificConsumptionScope::IdentifiedCausal,
                        &ExecutionContext::for_tests(319)
                    )
                    .is_err());
                let mut isolated = bindings.clone();
                isolated.components.hybrids.fill(components.hybrids[0]);
                isolated.stratum_worlds =
                    vec![isolated.components.hybrids, [ScientificWorldId::from_raw(8); 16]];
                let population_only = result
                    .decompose_population_reference(
                        &isolated,
                        &diagram,
                        ScientificConsumptionScope::IdentifiedCausal,
                        &ExecutionContext::for_tests(319),
                    )
                    .unwrap();
                assert!(
                    (population_only.decomposition.point.shapley[2]
                        - population_only.decomposition.point.total)
                        .abs()
                        < 1e-10
                );
                for other in [0, 1, 3] {
                    assert!(population_only.decomposition.point.shapley[other].abs() < 1e-10);
                }
            }
            let invalid =
                SelectionDiagram::try_new(graph.clone(), [VariableId::from_raw(4)]).unwrap();
            assert!(posterior
                .decompose_population_reference(
                    &bindings,
                    &invalid,
                    ScientificConsumptionScope::IdentifiedCausal,
                    &ExecutionContext::for_tests(319)
                )
                .is_err());
            graph.insert_bidirected(DenseNodeId::from_raw(3), DenseNodeId::from_raw(4)).unwrap();
            let mut fallback = components.clone();
            fallback.exposure = (0..4).map(VariableId::from_raw).collect();
            fallback.saturation.clear();
            fallback.mechanism.clear();
            fallback.hybrids = std::array::from_fn(|mask| {
                ScientificWorldId::from_raw(if mask & 1 == 0 { 0 } else { 15 })
            });
            let attempt = posterior
                .attempt_population_reference(
                    &bindings,
                    &invalid,
                    ScientificConsumptionScope::IdentifiedCausal,
                    &fallback,
                    &ExecutionContext::for_tests(319),
                )
                .unwrap();
            let PopulationReferenceAttempt::Coarser { refusal, executed } = attempt else {
                panic!("invalid population transport was accepted");
            };
            assert!(refusal.contains("selection"));
            assert!(executed.decomposition.point.total > 3.0);
            let confounded = SelectionDiagram::try_new(graph, [VariableId::from_raw(3)]).unwrap();
            assert!(posterior
                .decompose_population_reference(
                    &bindings,
                    &confounded,
                    ScientificConsumptionScope::IdentifiedCausal,
                    &ExecutionContext::for_tests(319)
                )
                .is_err());
        }
    }

    /// Complete data-generation, design identification, native F/B fitting and hybrid
    /// decomposition measurement. Execution does not grant a release calibration license.
    #[test]
    #[ignore = "Q08/E14 stable-candidate measurement; execute explicitly"]
    fn calibration_compiled_reference_decomposition_full_procedure() {
        let repetitions = 400;
        let truth_table: [f64; 16] = std::array::from_fn(|mask| {
            let a = if mask & 1 == 0 { 1.0 } else { 4.0 };
            let s = if mask & 2 == 0 { 1.0 } else { 2.0 };
            let m = if mask & 8 == 0 { 1.0 } else { 3.0 };
            (if mask & 4 == 0 { 1.2 } else { 1.8 }) * m * a / (a + s)
        });
        let truth: [f64; 4] = std::array::from_fn(|component| {
            (0_usize..16)
                .filter(|mask| mask & (1 << component) == 0)
                .map(|mask| {
                    let weight = match mask.count_ones() {
                        0 | 3 => 0.25,
                        1 | 2 => 1.0 / 12.0,
                        _ => unreachable!(),
                    };
                    weight * (truth_table[mask | (1 << component)] - truth_table[mask])
                })
                .sum()
        });
        for temporal in [false, true] {
            let mut coverage = [0; 2];
            let mut width = [0.0; 2];
            let mut ordering_sensitivity = [0.0; 2];
            for seed in 0..repetitions {
                let (posterior, bindings) = compiled_parameter_population_scenario(
                    temporal,
                    Some(0x28F100 + seed),
                    2048,
                    true,
                );
                let mut graph = antecedent_graph::Admg::with_variables(5);
                for variable in 0..4 {
                    graph
                        .insert_directed(
                            antecedent_graph::DenseNodeId::from_raw(variable),
                            antecedent_graph::DenseNodeId::from_raw(4),
                        )
                        .unwrap();
                }
                let diagram =
                    antecedent_graph::SelectionDiagram::try_new(graph, [VariableId::from_raw(3)])
                        .unwrap();
                let population = PopulationReferenceBindings {
                    baseline: VariableId::from_raw(3),
                    strata: vec![0.0, 1.0],
                    stratum_worlds: vec![
                        bindings.hybrids,
                        bindings.hybrids.map(|id| ScientificWorldId::from_raw(id.raw() + 8)),
                    ],
                    before_weights: vec![0.8, 0.2],
                    after_weights: vec![0.2, 0.8],
                    components: bindings.clone(),
                };
                let frequentist = posterior
                    .prepared()
                    .execute(JointDesignInference::Frequentist, &ExecutionContext::for_tests(seed))
                    .unwrap();
                let f = frequentist
                    .decompose_population_reference(
                        &population,
                        &diagram,
                        ScientificConsumptionScope::IdentifiedCausal,
                        &ExecutionContext::for_tests(seed),
                    )
                    .unwrap();
                let b = posterior
                    .decompose_population_reference(
                        &population,
                        &diagram,
                        ScientificConsumptionScope::IdentifiedCausal,
                        &ExecutionContext::for_tests(seed),
                    )
                    .unwrap();
                let mut simultaneous = [true; 2];
                for component in 0..4 {
                    let half_width = 2.49770547441237
                        * f.covariance.as_ref().unwrap().shapley[component][component]
                            .max(0.0)
                            .sqrt();
                    let point = f.decomposition.point.shapley[component];
                    simultaneous[0] &= truth[component] >= point - half_width
                        && truth[component] <= point + half_width;
                    width[0] += 2.0 * half_width;
                    ordering_sensitivity[0] += f.decomposition.point.order_range[component][1]
                        - f.decomposition.point.order_range[component][0];
                    ordering_sensitivity[1] += b.decomposition.point.order_range[component][1]
                        - b.decomposition.point.order_range[component][0];
                    let mut values = b
                        .decomposition
                        .draws
                        .iter()
                        .flatten()
                        .map(|draw| draw.shapley[component])
                        .collect::<Vec<_>>();
                    values.sort_by(f64::total_cmp);
                    let lower = values[12];
                    let upper = values[2035];
                    simultaneous[1] &= truth[component] >= lower && truth[component] <= upper;
                    width[1] += upper - lower;
                }
                for mode in 0..2 {
                    if simultaneous[mode] {
                        coverage[mode] += 1;
                    }
                }
            }
            for mode in 0..2 {
                let fraction = f64::from(coverage[mode]) / repetitions as f64;
                eprintln!(
                    "{}",
                    serde_json::json!({"procedure":"compiled_reference_decomposition_full_procedure","temporal":temporal,"inference":if mode==0 {"frequentist"} else {"bayesian"},"repetitions":repetitions,"simultaneous_coverage":fraction,"mean_component_width":width[mode]/(4*repetitions) as f64,"mean_ordering_sensitivity":ordering_sensitivity[mode]/(4*repetitions) as f64,"failed_slots":0,"nominal_joint_probability":0.95,"grants_release_license":false})
                );
                assert!(
                    fraction >= 0.90,
                    "joint coverage {fraction} at temporal={temporal}, mode={mode}"
                );
            }
        }
    }
}

impl crate::analysis::joint_panel::designs::JointDesignResult {
    /// Decompose actual design-native response worlds, retaining the complete scientific
    /// proof and the native joint sampling/posterior covariance. Shapley and sequential
    /// contributions are linear transformations of the full hybrid response vector;
    /// this propagates every off-diagonal covariance rather than combining marginal errors.
    /// # Errors
    /// Unsupported scope, missing/bad hybrids, covariance failure, resources or cancellation.
    pub fn decompose_reference_components(
        &self,
        bindings: &ReferenceComponentBindings,
        scope: ScientificConsumptionScope,
        ctx: &ExecutionContext,
    ) -> Result<ScientificReferenceDecomposition, JointPanelError> {
        let source = match self.law() {
            antecedent_estimate::joint_design::compiled::JointDesignLaw::Bayesian(_) => {
                self.posterior_source()?
            }
            antecedent_estimate::joint_design::compiled::JointDesignLaw::Frequentist(_)
            | antecedent_estimate::joint_design::compiled::JointDesignLaw::FrequentistBootstrap {
                ..
            } => self.mean_source()?,
        };
        let mut result = decompose_scientific_reference(&source, bindings, scope, ctx)?;
        let worlds = &self.prepared().proof().worlds;
        let mut indices = [0; 16];
        for (mask, id) in bindings.hybrids.iter().enumerate() {
            indices[mask] =
                worlds.iter().position(|world| world.world == id.raw()).ok_or_else(|| {
                    invalid("hybrid absent from original design-functional covariance")
                })?;
        }
        if let Some(realizations) = self.law().frequentist_realizations() {
            let bytes = realizations
                .len()
                .checked_mul(16 * 8 + 16)
                .ok_or_else(|| invalid("bootstrap reference projection quote overflow"))?;
            let _projection = ctx
                .acquire_admission(0, bytes as u64)
                .map_err(|error| super::engine::PanelError::Resource(error.to_string()))?;
            let draws = realizations
                .iter()
                .map(|row| {
                    let row = row.as_ref()?;
                    if row.len() != worlds.len() {
                        return None;
                    }
                    Some(indices.map(|index| row[index]))
                })
                .collect::<Vec<_>>();
            result.decomposition = super::engine::decomposition::decompose_hybrid_law(
                result.decomposition.point.hybrids,
                &draws,
                bindings.order,
                ctx,
            )?;
            result.procedure_failures =
                self.law().frequentist_failures().unwrap_or_default().to_vec();
        }
        let native = self.law().estimate();
        if native.covariance.is_empty() && self.law().frequentist_realizations().is_some() {
            return Ok(result);
        }
        if native.covariance.len() != worlds.len() * worlds.len() {
            return Err(invalid("native joint-design covariance axis mismatch"));
        }
        let covariance = std::array::from_fn(|a| {
            std::array::from_fn(|b| native.covariance[indices[a] * worlds.len() + indices[b]])
        });
        result.covariance = Some(super::engine::decomposition::propagate_hybrid_covariance(
            &covariance,
            bindings.order,
        )?);
        Ok(result)
    }
}

mod artifact;
pub use artifact::{
    replay_population_reference, replay_reference_decomposition, HistoricalPopulationReference,
    HistoricalReferenceDecomposition,
};
mod population;
pub use population::{
    decompose_population_reference, PopulationReferenceAttempt, PopulationReferenceBindings,
    PopulationReferenceDecomposition, PopulationReferenceTransportProof,
};
