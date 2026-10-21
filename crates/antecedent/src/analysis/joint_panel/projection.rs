//! Exact prefix-world basis for downstream temporal response functionals.
// SPDX-License-Identifier: MIT OR Apache-2.0
use super::{
    JointPanelError, JointPanelFunctional, JointPanelResult, JointPanelWorld, contracts, engine,
    joint_panel_target, source,
};
use antecedent_contract::ScientificQuantity;
use antecedent_core::{Assumption, ModelAssumptions, ScientificTarget, ScientificWorldId};
use antecedent_design::decision_eval::MeanSource;
use antecedent_design::scientific_source::{
    ScientificDecisionSource, ScientificProjectionBasis, ScientificProjectionOutput,
    ScientificSourcePayload, ScientificSourceProjection,
};
use antecedent_io::distribution_artifact::DistributionArtifact;

impl JointPanelResult {
    pub(super) fn projected_scientific_source(
        &self,
        payload: ScientificSourcePayload,
        weights: Option<Vec<Vec<f64>>>,
    ) -> Result<ScientificDecisionSource, JointPanelError> {
        self.projected_scientific_source_with_dependency(
            payload,
            weights,
            self.dependency_identity(),
        )
    }
    pub(super) fn projected_scientific_source_with_dependency(
        &self,
        payload: ScientificSourcePayload,
        weights: Option<Vec<Vec<f64>>>,
        dependency: antecedent_core::SemanticDigest,
    ) -> Result<ScientificDecisionSource, JointPanelError> {
        let borrowed = match &payload {
            ScientificSourcePayload::Means(means) => ProjectionPayloadRef::Means(means),
            ScientificSourcePayload::Law(law) => ProjectionPayloadRef::Law(law),
        };
        let bytes = projection_quote(self, borrowed)?;
        self.projected_source_admitted(|| payload, weights, dependency, bytes)
    }
    pub(super) fn projected_law_source_with_dependency(
        &self,
        law: &DistributionArtifact,
        weights: Option<Vec<Vec<f64>>>,
        dependency: antecedent_core::SemanticDigest,
    ) -> Result<ScientificDecisionSource, JointPanelError> {
        let bytes = projection_quote(self, ProjectionPayloadRef::Law(law))?;
        self.projected_source_admitted(
            || ScientificSourcePayload::Law(Box::new(law.clone())),
            weights,
            dependency,
            bytes,
        )
    }
    fn projected_source_admitted(
        &self,
        payload: impl FnOnce() -> ScientificSourcePayload,
        weights: Option<Vec<Vec<f64>>>,
        dependency: antecedent_core::SemanticDigest,
        bytes: u64,
    ) -> Result<ScientificDecisionSource, JointPanelError> {
        source::check_cancelled(&self.context)?;
        // Prefix schedules/context have quadratic retained width in the horizon. Admit
        // that real expansion and the numerical clone before allocating either.
        source::check_memory(&self.context, bytes)?;
        let lease = self
            .context
            .acquire_admission(0, bytes)
            .map_err(|error| engine::PanelError::Resource(error.to_string()))?;
        let payload = payload();
        let (target, basis) = self.projection_basis()?;
        let output_coordinates = match &payload {
            ScientificSourcePayload::Means(means) => means.coordinates.clone(),
            ScientificSourcePayload::Law(law) => law
                .quantities()
                .iter()
                .cloned()
                .map(ScientificQuantity::try_from)
                .collect::<Result<Vec<_>, _>>()
                .map_err(JointPanelError::compile)?,
        };
        let rows = if let Some(weights) = weights {
            let mut offset = 0;
            weights
                .into_iter()
                .map(|weights| {
                    let width = weights.len();
                    let terms = weights
                        .into_iter()
                        .enumerate()
                        .filter_map(|(index, coefficient)| {
                            (coefficient != 0.0).then_some((offset + index, coefficient))
                        })
                        .collect::<Vec<_>>();
                    offset += width;
                    terms
                })
                .collect::<Vec<_>>()
        } else {
            (0..basis.len()).map(|index| vec![(index, 1.0)]).collect()
        };
        let projection = ScientificSourceProjection {
            basis,
            outputs: output_coordinates
                .into_iter()
                .zip(rows)
                .map(|(quantity, terms)| ScientificProjectionOutput { quantity, terms })
                .collect(),
        };
        ScientificDecisionSource::new_projected(
            target.clone(),
            contracts::conditional_target_product(&target)?,
            dependency,
            payload,
            projection,
        )
        .map(|source| source.retain_admission(std::sync::Arc::new(lease)))
        .map_err(JointPanelError::compile)
    }
    fn projection_basis(
        &self,
    ) -> Result<(ScientificTarget, Vec<ScientificProjectionBasis>), JointPanelError> {
        let coordinates = self.coordinates()?;
        let mut prefixes = Vec::<JointPanelWorld>::new();
        let mut basis_coordinates = Vec::new();
        for original in &self.request().worlds {
            for period in 0..original.horizon() {
                source::check_cancelled(&self.context)?;
                let mut prefix = original.clone();
                prefix.world = ScientificWorldId::from_raw(
                    u32::try_from(prefixes.len()).map_err(JointPanelError::compile)?,
                );
                prefix.reference = prefixes.is_empty();
                prefix.program.id = format!("{}@period={period}", original.program.id);
                for channel in &mut prefix.program.channels {
                    channel.values.truncate(period + 1);
                }
                for scenario in &mut prefix.program.scenarios {
                    scenario.confounder.truncate(period + 1);
                    scenario.season.truncate(period + 1);
                }
                basis_coordinates.push(coordinates[prefixes.len()].clone());
                prefixes.push(prefix);
            }
        }
        let generated = joint_panel_target(
            self.prepared().source_data().schema(),
            &self.request().columns,
            self.request().lag_order,
            &prefixes,
            JointPanelFunctional::Worlds,
            self.request().target.inference(),
        )?;
        let mut assumptions = generated.assumptions().records().clone();
        // Original context certificates describe whole programs. Prefix context laws
        // are generated above; retain every other scientific premise unchanged.
        assumptions.extend_unique(self.request().target.assumptions().records().entries.iter().filter(|record| !matches!(&record.assumption, Assumption::Custom{id,..} if id.starts_with("panel.context.world."))));
        let generated_worlds = contracts::target_worlds(&generated)?;
        let mut temporal_worlds = Vec::with_capacity(generated_worlds.len());
        let mut index = 0;
        for original in &self.request().worlds {
            let source_world = contracts::target_worlds(&self.request().target)?
                .iter()
                .find(|world| world.id() == original.world)
                .ok_or_else(|| JointPanelError::compile("missing original prefix world"))?;
            for period in 0..original.horizon() {
                let world = &generated_worlds[index];
                temporal_worlds.push(restore_prefix_clock(
                    world,
                    source_world,
                    u32::try_from(period + 1).map_err(JointPanelError::compile)?,
                )?);
                index += 1;
            }
        }
        let target = ScientificTarget::new(
            antecedent_core::TargetForm::Worlds(
                antecedent_core::WorldCollection::new(temporal_worlds)
                    .map_err(JointPanelError::compile)?,
            ),
            ModelAssumptions::new(self.request().target.assumptions().graph(), assumptions),
            generated.inference(),
        );
        let worlds = contracts::target_worlds(&target)?;
        let basis = worlds
            .iter()
            .zip(basis_coordinates)
            .map(|(world, mut quantity)| {
                quantity.regime_id =
                    format!("compiled_world:{}:{}", target.identity(), world.id().raw());
                quantity.horizon = world.time().horizon_steps().unwrap_or(0);
                ScientificProjectionBasis {
                    world: world.id(),
                    variable: self.request().columns.outcome,
                    quantity,
                }
            })
            .collect::<Vec<_>>();
        Ok((target, basis))
    }
}
/// Preserve the source world's clock, history and initial law in a shorter actual program.
pub(super) fn restore_prefix_clock(
    prefix: &antecedent_core::InterventionWorld,
    original: &antecedent_core::InterventionWorld,
    horizon: u32,
) -> Result<antecedent_core::InterventionWorld, JointPanelError> {
    let time = if original.time().is_static() {
        prefix.time().clone()
    } else {
        antecedent_core::TimeFrame::temporal(
            horizon,
            original
                .time()
                .history()
                .ok_or_else(|| JointPanelError::compile("missing original history"))?,
            original
                .time()
                .initial_law()
                .ok_or_else(|| JointPanelError::compile("missing original initial law"))?
                .clone(),
        )
        .map_err(JointPanelError::compile)?
    };
    let action = if time.is_static() {
        prefix.action().clone()
    } else if let Some(interventions) = prefix.action().at_decision_interventions() {
        antecedent_core::WorldAction::schedule(
            antecedent_core::Intervention::Sequence(antecedent_core::InterventionSequence::new(
                interventions
                    .iter()
                    .cloned()
                    .map(|intervention| {
                        antecedent_core::SequencedIntervention::new(
                            intervention,
                            antecedent_core::TemporalPolicy::pulse(0),
                        )
                    })
                    .collect::<Vec<_>>(),
            )),
            antecedent_core::TemporalPolicy::sustained(0, 0),
        )
        .map_err(JointPanelError::compile)?
    } else {
        prefix.action().clone()
    };
    antecedent_core::InterventionWorld::new(
        prefix.id(),
        action,
        prefix.is_reference(),
        prefix.binding().clone(),
        time,
    )
    .map_err(JointPanelError::compile)
}

#[derive(Clone, Copy)]
enum ProjectionPayloadRef<'a> {
    Means(&'a MeanSource),
    Law(&'a DistributionArtifact),
}
fn projection_quote(
    result: &JointPanelResult,
    payload: ProjectionPayloadRef<'_>,
) -> Result<u64, JointPanelError> {
    let overflow =
        || engine::PanelError::Resource("prefix target projection width overflow".into());
    let mut bytes = 0_u64;
    let mut add = |value: usize| -> Result<(), JointPanelError> {
        bytes = bytes
            .checked_add(u64::try_from(value).map_err(|_| overflow())?)
            .ok_or_else(overflow)?;
        Ok(())
    };
    let request = result.request();
    for world in &request.worlds {
        let h = world.horizon();
        // Cloning then truncating preserves the original vector capacity. Include
        // both the temporary full-capacity program and encoded per-step target.
        let per_period = request
            .columns
            .channels
            .len()
            .checked_mul(512)
            .and_then(|v| v.checked_add(world.program.scenarios.len().checked_mul(64)?))
            .ok_or_else(overflow)?;
        let mut fixed = world.program.id.len().checked_add(4096).ok_or_else(overflow)?;
        for scenario in &world.program.scenarios {
            fixed = fixed
                .checked_add(scenario.group.len())
                .and_then(|value| value.checked_add(256))
                .ok_or_else(overflow)?;
            for history in &scenario.history {
                fixed = fixed
                    .checked_add(history.id.len())
                    .and_then(|value| value.checked_add(64))
                    .ok_or_else(overflow)?;
            }
        }
        add(h
            .checked_mul(
                fixed
                    .checked_add(h.checked_mul(per_period).ok_or_else(overflow)?)
                    .ok_or_else(overflow)?,
            )
            .ok_or_else(overflow)?)?;
    }
    let basis_count = request.worlds.iter().try_fold(0_usize, |count, world| {
        count.checked_add(world.horizon()).ok_or_else(overflow)
    })?;
    u32::try_from(basis_count).map_err(|_| overflow())?;
    for variable in result.prepared().source_data().schema().variables() {
        add(variable
            .name
            .len()
            .checked_add(variable.unit.as_ref().map_or(0, |x| x.len()))
            .and_then(|v| v.checked_mul(basis_count)?.checked_mul(8))
            .ok_or_else(overflow)?)?;
    }
    add(request
        .target
        .assumptions()
        .records()
        .len()
        .checked_mul(std::mem::size_of::<antecedent_core::AssumptionRecord>())
        .ok_or_else(overflow)?)?;
    add(payload_quote(payload)?)?;
    Ok(bytes)
}
pub(super) fn source_payload_quote(
    payload: &ScientificSourcePayload,
) -> Result<u64, JointPanelError> {
    let borrowed = match payload {
        ScientificSourcePayload::Means(means) => ProjectionPayloadRef::Means(means),
        ScientificSourcePayload::Law(law) => ProjectionPayloadRef::Law(law),
    };
    u64::try_from(payload_quote(borrowed)?)
        .map_err(|_| engine::PanelError::Resource("source payload width overflow".into()).into())
}
fn payload_quote(payload: ProjectionPayloadRef<'_>) -> Result<usize, JointPanelError> {
    let overflow = || engine::PanelError::Resource("projected payload width overflow".into());
    let mut bytes = 0_usize;
    let mut add = |value: usize| -> Result<(), JointPanelError> {
        bytes = bytes.checked_add(value).ok_or_else(overflow)?;
        Ok(())
    };
    match payload {
        ProjectionPayloadRef::Means(means) => {
            add(means.means.len().checked_mul(8).ok_or_else(overflow)?)?;
            for q in &means.coordinates {
                add(quantity_bytes(q)?.checked_mul(4).ok_or_else(overflow)?)?;
            }
        }
        ProjectionPayloadRef::Law(law) => {
            add(law.draws().len().checked_mul(8).ok_or_else(overflow)?)?;
            let metadata = law.metadata();
            if metadata.legacy_posterior.is_some() || metadata.legacy_bindings.is_some() {
                return Err(JointPanelError::Unsupported(
                    "native panel law cannot import a legacy posterior carrier".into(),
                ));
            }
            add(std::mem::size_of_val(law))?;
            for text in [
                &metadata.identity.source_id,
                &metadata.identity.provider_id,
                &metadata.identity.rng_id,
                &metadata.identity.snapshot_id,
                &metadata.identity.causal_contract_id,
                &metadata.axes[0],
                &metadata.axes[1],
            ] {
                add(text.len())?;
            }
            add(metadata
                .weights
                .as_ref()
                .map_or(0, Vec::len)
                .checked_mul(8)
                .ok_or_else(overflow)?)?;
            add(metadata.supported.as_ref().map_or(0, Vec::len))?;
            for q in law.quantities() {
                add(wire_quantity_bytes(q)?.checked_mul(4).ok_or_else(overflow)?)?;
            }
        }
    }
    Ok(bytes)
}
fn quantity_bytes(q: &ScientificQuantity) -> Result<usize, JointPanelError> {
    let mut bytes = std::mem::size_of::<ScientificQuantity>();
    for value in [
        &q.variable_id,
        &q.variable_name,
        &q.units,
        &q.population_id,
        &q.regime_id,
        &q.functional_id,
        &q.transform_id,
    ] {
        bytes = bytes
            .checked_add(value.len())
            .ok_or_else(|| engine::PanelError::Resource("quantity width overflow".into()))?;
    }
    for value in &q.conditioning {
        bytes = bytes
            .checked_add(
                value.variable_id.len() + value.value_id.len() + std::mem::size_of_val(value),
            )
            .ok_or_else(|| engine::PanelError::Resource("conditioning width overflow".into()))?;
    }
    Ok(bytes)
}

fn wire_quantity_bytes(
    q: &antecedent_io::quantity_wire::ScientificQuantityWire,
) -> Result<usize, JointPanelError> {
    let overflow = || engine::PanelError::Resource("quantity width overflow".into());
    let mut bytes = std::mem::size_of_val(q);
    for value in [
        &q.variable_id,
        &q.variable_name,
        &q.role,
        &q.units,
        &q.population_id,
        &q.regime_id,
        &q.functional_id,
        &q.transform_id,
    ] {
        bytes = bytes.checked_add(value.len()).ok_or_else(overflow)?;
    }
    for value in &q.conditioning {
        bytes = bytes
            .checked_add(std::mem::size_of_val(value))
            .and_then(|v| v.checked_add(value.variable_id.len()))
            .and_then(|v| v.checked_add(value.value_id.len()))
            .ok_or_else(overflow)?;
    }
    Ok(bytes)
}
