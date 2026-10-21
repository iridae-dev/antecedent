//! Versioned scientific targets shared by artifacts and language bindings.
//!
//! Decoding rebuilds the validated domain constructors and checks the complete target
//! identity. This transports a question; it grants no identification or execution license.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::query_wire::OutcomeFunctionalWire;
use crate::trace::{AssumptionRecordWire, assumptions_from_wire, assumptions_to_wire};
use crate::{InterventionWire, IoError, TargetPopulationWire, TemporalPolicyWire};
use antecedent_core::{
    Conditioning, ContrastForm, CrossWorldPremise, DoseTimeSurface, GraphFamily, HistoryWindow,
    InferenceSelection, InitialLaw, InterventionContrast, InterventionWorld, MediationArms,
    MediationComponent, MediationTarget, ModelAssumptions, ObservationUnit, OutcomeQuantity,
    QuantityUnits, ScientificTarget, ScientificWorldId, TargetForm, TimeFrame, VariableId,
    WorldAction, WorldBinding, WorldCollection,
};
use serde::{Deserialize, Serialize};

/// Current scientific-target wire version.
pub const SCIENTIFIC_TARGET_WIRE_VERSION: u16 = 2;

/// Complete portable question, including its inference realization and checked identity.
///
/// Private fields ensure callers construct it from a validated domain target. Untrusted
/// decoded bytes still require [`Self::to_domain`]; a digest alone never authorizes a claim.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ScientificTargetWire {
    version: u16,
    identity: [u8; 32],
    graph: String,
    assumptions: Vec<AssumptionRecordWire>,
    inference: String,
    target: FormWire,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum FormWire {
    Linear {
        worlds: Vec<WorldWire>,
        terms: Vec<(u32, u64)>,
        reference: u32,
    },
    World {
        world: Box<WorldWire>,
    },
    Worlds {
        worlds: Vec<WorldWire>,
    },
    Difference {
        worlds: Vec<WorldWire>,
        plus: u32,
        minus: u32,
        reference: u32,
    },
    Interaction {
        worlds: Vec<WorldWire>,
        joint: u32,
        a_at_reference: u32,
        b_at_reference: u32,
        reference: u32,
    },
    DoseTime {
        worlds: Vec<WorldWire>,
        dose: u32,
        levels: Vec<u64>,
        horizons: Vec<u32>,
        cells: Vec<u32>,
    },
    Mediation {
        binding: Box<BindingWire>,
        time: TimeWire,
        schedule: Option<TemporalPolicyWire>,
        treatment: u32,
        mediators: Vec<u32>,
        control: InterventionWire,
        active: InterventionWire,
        component: String,
        premises: Vec<String>,
    },
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct BindingWire {
    population: TargetPopulationWire,
    observation_unit: String,
    outcome: u32,
    functional: OutcomeFunctionalWire,
    units: String,
    conditioning: Vec<u32>,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum TimeWire {
    Static,
    Temporal { horizon: u32, max_lag: Option<u32>, initial_law: Option<String> },
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum ActionWire {
    Simultaneous { interventions: Vec<InterventionWire> },
    Scheduled { intervention: InterventionWire, policy: TemporalPolicyWire },
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct WorldWire {
    id: u32,
    reference: bool,
    binding: BindingWire,
    time: TimeWire,
    action: ActionWire,
}

fn invalid(error: impl std::fmt::Display) -> IoError {
    IoError::Convert(format!("scientific_target: {error}"))
}
fn unknown(field: &str) -> IoError {
    invalid(format!("unsupported {field}"))
}

impl ScientificTargetWire {
    /// Encode a complete question without fitting or identifying it.
    ///
    /// # Errors
    /// Unsupported future variants or populations requiring an external registry.
    pub fn from_domain(target: &ScientificTarget) -> Result<Self, IoError> {
        let graph = match target.assumptions().graph() {
            GraphFamily::Unspecified => "unspecified",
            GraphFamily::Dag => "dag",
            GraphFamily::Admg => "admg",
            GraphFamily::Cpdag => "cpdag",
            GraphFamily::Pag => "pag",
            GraphFamily::TemporalDag => "temporal_dag",
            GraphFamily::TemporalCpdag => "temporal_cpdag",
            GraphFamily::TemporalPag => "temporal_pag",
            GraphFamily::TemporalAdmg => "temporal_admg",
            GraphFamily::CoDetermined => "co_determined",
            GraphFamily::Graphless => "graphless",
            _ => return Err(unknown("graph family")),
        }
        .to_owned();
        let inference = match target.inference() {
            InferenceSelection::Unspecified => "unspecified",
            InferenceSelection::Frequentist => "frequentist",
            InferenceSelection::Bayesian => "bayesian",
            InferenceSelection::GeneralizedBayesian => "generalized_bayesian",
            _ => return Err(unknown("inference")),
        }
        .to_owned();
        let form = FormWire::from_domain(target.form())?;
        Ok(Self {
            version: SCIENTIFIC_TARGET_WIRE_VERSION,
            identity: *target.identity().as_bytes(),
            graph,
            assumptions: assumptions_to_wire(target.assumptions().records()),
            inference,
            target: form,
        })
    }

    /// Rebuild the question through its domain validation and verify its identity.
    ///
    /// # Errors
    /// Unknown versions, malformed worlds, inconsistent contrasts, missing mediation
    /// premises, unsupported field values, or a changed scientific identity.
    pub fn to_domain(&self) -> Result<ScientificTarget, IoError> {
        if !matches!(self.version, 1 | SCIENTIFIC_TARGET_WIRE_VERSION)
            || (self.version == 1 && matches!(self.target, FormWire::Linear { .. }))
        {
            return Err(IoError::UnsupportedVersion { version: u32::from(self.version) });
        }
        let graph = match self.graph.as_str() {
            "unspecified" => GraphFamily::Unspecified,
            "dag" => GraphFamily::Dag,
            "admg" => GraphFamily::Admg,
            "cpdag" => GraphFamily::Cpdag,
            "pag" => GraphFamily::Pag,
            "temporal_dag" => GraphFamily::TemporalDag,
            "temporal_cpdag" => GraphFamily::TemporalCpdag,
            "temporal_pag" => GraphFamily::TemporalPag,
            "temporal_admg" => GraphFamily::TemporalAdmg,
            "co_determined" => GraphFamily::CoDetermined,
            "graphless" => GraphFamily::Graphless,
            _ => return Err(unknown("graph family")),
        };
        let inference = match self.inference.as_str() {
            "unspecified" => InferenceSelection::Unspecified,
            "frequentist" => InferenceSelection::Frequentist,
            "bayesian" => InferenceSelection::Bayesian,
            "generalized_bayesian" => InferenceSelection::GeneralizedBayesian,
            _ => return Err(unknown("inference")),
        };
        let form = self.target.to_domain()?;
        let target = ScientificTarget::new(
            form,
            ModelAssumptions::new(graph, assumptions_from_wire(&self.assumptions)?),
            inference,
        );
        if *target.identity().as_bytes() != self.identity {
            return Err(invalid("identity mismatch"));
        }
        Ok(target)
    }
}

fn decode_worlds(worlds: &[WorldWire]) -> Result<WorldCollection, IoError> {
    WorldCollection::new(worlds.iter().map(WorldWire::to_domain).collect::<Result<Vec<_>, _>>()?)
        .map_err(invalid)
}

impl FormWire {
    fn from_domain(form: &TargetForm) -> Result<Self, IoError> {
        let worlds = |collection: &WorldCollection| {
            collection.worlds().iter().map(WorldWire::from_domain).collect::<Result<Vec<_>, _>>()
        };
        Ok(match form {
            TargetForm::World(world) => {
                FormWire::World { world: Box::new(WorldWire::from_domain(world)?) }
            }
            TargetForm::Worlds(collection) => FormWire::Worlds { worlds: worlds(collection)? },
            TargetForm::Contrast(contrast) => match contrast.form() {
                ContrastForm::Linear => FormWire::Linear {
                    worlds: worlds(contrast.worlds())?,
                    terms: contrast
                        .terms()
                        .iter()
                        .map(|term| (term.world.raw(), term.coefficient.to_bits()))
                        .collect(),
                    reference: contrast.reference().raw(),
                },
                ContrastForm::Difference { plus, minus } => FormWire::Difference {
                    worlds: worlds(contrast.worlds())?,
                    plus: plus.raw(),
                    minus: minus.raw(),
                    reference: contrast.reference().raw(),
                },
                ContrastForm::Interaction { joint, a_at_reference, b_at_reference } => {
                    FormWire::Interaction {
                        worlds: worlds(contrast.worlds())?,
                        joint: joint.raw(),
                        a_at_reference: a_at_reference.raw(),
                        b_at_reference: b_at_reference.raw(),
                        reference: contrast.reference().raw(),
                    }
                }
                _ => return Err(unknown("contrast")),
            },
            TargetForm::DoseTime(surface) => FormWire::DoseTime {
                worlds: worlds(surface.worlds())?,
                dose: surface.dose().raw(),
                levels: surface.level_bits().to_vec(),
                horizons: surface.horizons().to_vec(),
                cells: surface.cells().iter().map(|id| id.raw()).collect(),
            },
            TargetForm::Mediation(mediation) => {
                let arms = mediation.arms();
                let component = match arms.component() {
                    MediationComponent::Total => "total",
                    MediationComponent::Direct => "direct",
                    MediationComponent::Mediated => "mediated",
                    MediationComponent::NaturalDirect => "natural_direct",
                    MediationComponent::NaturalIndirect => "natural_indirect",
                    _ => return Err(unknown("mediation component")),
                }
                .to_owned();
                let premises = mediation
                    .premises()
                    .iter()
                    .map(|premise| {
                        Ok(match premise {
                            CrossWorldPremise::Composition => "composition",
                            CrossWorldPremise::CrossWorldIndependence => "cross_world_independence",
                            CrossWorldPremise::PathSpecificNested => "path_specific_nested",
                            _ => return Err(unknown("cross-world premise")),
                        }
                        .to_owned())
                    })
                    .collect::<Result<Vec<_>, IoError>>()?;
                FormWire::Mediation {
                    binding: Box::new(BindingWire::from_domain(mediation.binding())?),
                    time: TimeWire::from_domain(mediation.time())?,
                    schedule: mediation
                        .schedule()
                        .map(TemporalPolicyWire::from_domain)
                        .transpose()?,
                    treatment: arms.treatment().raw(),
                    mediators: arms.mediators().iter().map(|id| id.raw()).collect(),
                    control: InterventionWire::from_domain(arms.control())?,
                    active: InterventionWire::from_domain(arms.active())?,
                    component,
                    premises,
                }
            }
            _ => return Err(unknown("target form")),
        })
    }
    fn to_domain(&self) -> Result<TargetForm, IoError> {
        let id = ScientificWorldId::from_raw;
        Ok(match self {
            FormWire::Linear { worlds: items, terms, reference } => TargetForm::Contrast(
                InterventionContrast::linear(
                    decode_worlds(items)?,
                    terms.iter().map(|(world, bits)| (id(*world), f64::from_bits(*bits))),
                    id(*reference),
                )
                .map_err(invalid)?,
            ),
            FormWire::World { world } => TargetForm::World(world.to_domain()?),
            FormWire::Worlds { worlds: items } => TargetForm::Worlds(decode_worlds(items)?),
            FormWire::Difference { worlds: items, plus, minus, reference } => TargetForm::Contrast(
                InterventionContrast::difference(
                    decode_worlds(items)?,
                    id(*plus),
                    id(*minus),
                    id(*reference),
                )
                .map_err(invalid)?,
            ),
            FormWire::Interaction {
                worlds: items,
                joint,
                a_at_reference,
                b_at_reference,
                reference,
            } => TargetForm::Contrast(
                InterventionContrast::interaction(
                    decode_worlds(items)?,
                    id(*joint),
                    id(*a_at_reference),
                    id(*b_at_reference),
                    id(*reference),
                )
                .map_err(invalid)?,
            ),
            FormWire::DoseTime { worlds: items, dose, levels, horizons, cells } => {
                TargetForm::DoseTime(
                    DoseTimeSurface::new(
                        decode_worlds(items)?,
                        VariableId::from_raw(*dose),
                        levels.iter().map(|bits| f64::from_bits(*bits)),
                        horizons.iter().copied(),
                        cells.iter().map(|raw| id(*raw)),
                    )
                    .map_err(invalid)?,
                )
            }
            FormWire::Mediation {
                binding,
                time,
                schedule,
                treatment,
                mediators,
                control,
                active,
                component,
                premises,
            } => {
                let component = match component.as_str() {
                    "total" => MediationComponent::Total,
                    "direct" => MediationComponent::Direct,
                    "mediated" => MediationComponent::Mediated,
                    "natural_direct" => MediationComponent::NaturalDirect,
                    "natural_indirect" => MediationComponent::NaturalIndirect,
                    _ => return Err(unknown("mediation component")),
                };
                let premises = premises
                    .iter()
                    .map(|premise| match premise.as_str() {
                        "composition" => Ok(CrossWorldPremise::Composition),
                        "cross_world_independence" => Ok(CrossWorldPremise::CrossWorldIndependence),
                        "path_specific_nested" => Ok(CrossWorldPremise::PathSpecificNested),
                        _ => Err(unknown("cross-world premise")),
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let arms = MediationArms::new(
                    VariableId::from_raw(*treatment),
                    mediators.iter().map(|raw| VariableId::from_raw(*raw)),
                    control.to_domain(),
                    active.to_domain(),
                    component,
                )
                .map_err(invalid)?;
                TargetForm::Mediation(
                    MediationTarget::new(
                        binding.to_domain()?,
                        time.to_domain()?,
                        schedule.as_ref().map(TemporalPolicyWire::to_domain),
                        arms,
                        premises,
                    )
                    .map_err(invalid)?,
                )
            }
        })
    }
}

impl BindingWire {
    fn from_domain(binding: &WorldBinding) -> Result<Self, IoError> {
        Ok(Self {
            population: TargetPopulationWire::from_domain(binding.population())?,
            observation_unit: binding.unit().name().into(),
            outcome: binding.outcome().outcome().raw(),
            functional: OutcomeFunctionalWire::from_domain(binding.outcome().functional()),
            units: binding.outcome().units().name().into(),
            conditioning: binding.conditioning().variables().iter().map(|id| id.raw()).collect(),
        })
    }
    fn to_domain(&self) -> Result<WorldBinding, IoError> {
        WorldBinding::new(
            self.population.to_domain()?,
            ObservationUnit::new(self.observation_unit.as_str()).map_err(invalid)?,
            OutcomeQuantity::new(
                VariableId::from_raw(self.outcome),
                self.functional.to_domain(),
                QuantityUnits::new(self.units.as_str()).map_err(invalid)?,
            )
            .map_err(invalid)?,
            Conditioning::on(self.conditioning.iter().map(|id| VariableId::from_raw(*id))),
        )
        .map_err(invalid)
    }
}
impl TimeWire {
    fn from_domain(time: &TimeFrame) -> Result<Self, IoError> {
        if time.is_static() {
            return Ok(Self::Static);
        }
        Ok(Self::Temporal {
            horizon: time.horizon_steps().ok_or_else(|| unknown("temporal horizon"))?,
            max_lag: time.history().ok_or_else(|| unknown("history"))?.max_lag_steps(),
            initial_law: time
                .initial_law()
                .ok_or_else(|| unknown("initial law"))?
                .name()
                .map(str::to_owned),
        })
    }
    fn to_domain(&self) -> Result<TimeFrame, IoError> {
        match self {
            Self::Static => Ok(TimeFrame::static_decision()),
            Self::Temporal { horizon, max_lag, initial_law } => TimeFrame::temporal(
                *horizon,
                max_lag.map_or_else(HistoryWindow::planner_default, HistoryWindow::max_lag),
                initial_law
                    .as_deref()
                    .map_or_else(|| Ok(InitialLaw::contemporaneous_observed()), InitialLaw::named)
                    .map_err(invalid)?,
            )
            .map_err(invalid),
        }
    }
}
impl WorldWire {
    fn from_domain(world: &InterventionWorld) -> Result<Self, IoError> {
        let action = if let Some(interventions) = world.action().at_decision_interventions() {
            ActionWire::Simultaneous {
                interventions: interventions
                    .iter()
                    .map(InterventionWire::from_domain)
                    .collect::<Result<Vec<_>, _>>()?,
            }
        } else {
            let (intervention, policy) =
                world.action().schedule_parts().ok_or_else(|| unknown("world action"))?;
            ActionWire::Scheduled {
                intervention: InterventionWire::from_domain(intervention)?,
                policy: TemporalPolicyWire::from_domain(policy)?,
            }
        };
        Ok(Self {
            id: world.id().raw(),
            reference: world.is_reference(),
            binding: BindingWire::from_domain(world.binding())?,
            time: TimeWire::from_domain(world.time())?,
            action,
        })
    }
    fn to_domain(&self) -> Result<InterventionWorld, IoError> {
        let action = match &self.action {
            ActionWire::Simultaneous { interventions } => {
                WorldAction::at_decision(interventions.iter().map(InterventionWire::to_domain))
                    .map_err(invalid)?
            }
            ActionWire::Scheduled { intervention, policy } => {
                WorldAction::schedule(intervention.to_domain(), policy.to_domain())
                    .map_err(invalid)?
            }
        };
        InterventionWorld::new(
            ScientificWorldId::from_raw(self.id),
            action,
            self.reference,
            self.binding.to_domain()?,
            self.time.to_domain()?,
        )
        .map_err(invalid)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_core::{AssumptionSet, Intervention, TargetPopulation, TemporalPolicy, Value};

    fn binding() -> WorldBinding {
        WorldBinding::new(
            TargetPopulation::AllObserved,
            ObservationUnit::new("geographic group").unwrap(),
            OutcomeQuantity::mean(VariableId::from_raw(2), QuantityUnits::new("USD").unwrap())
                .unwrap(),
            Conditioning::on([VariableId::from_raw(3)]),
        )
        .unwrap()
    }
    fn world(raw: u32, a: f64, b: f64, reference: bool) -> InterventionWorld {
        InterventionWorld::new(
            ScientificWorldId::from_raw(raw),
            WorldAction::at_decision([
                Intervention::set(VariableId::from_raw(0), Value::Float64(a)),
                Intervention::set(VariableId::from_raw(1), Value::Float64(b)),
            ])
            .unwrap(),
            reference,
            binding(),
            TimeFrame::static_decision(),
        )
        .unwrap()
    }
    fn target(form: TargetForm) -> ScientificTarget {
        ScientificTarget::new(
            form,
            ModelAssumptions::new(GraphFamily::Dag, AssumptionSet::new()),
            InferenceSelection::Bayesian,
        )
    }
    fn roundtrip(target: &ScientificTarget) {
        let wire = ScientificTargetWire::from_domain(target).unwrap();
        let bytes = crate::to_cbor(&wire).unwrap();
        let decoded: ScientificTargetWire = crate::from_cbor(&bytes).unwrap();
        let actual = decoded.to_domain().unwrap();
        assert_eq!(actual.identity(), target.identity());
        assert_eq!(actual.inference(), target.inference());
        assert_eq!(&actual, target);
    }
    #[test]
    fn scientific_target_wire_roundtrips_world_collection_and_interaction() {
        let worlds = WorldCollection::new([
            world(0, 0., 0., true),
            world(1, 4., 0., false),
            world(2, 0., 4., false),
            world(3, 4., 4., false),
        ])
        .unwrap();
        roundtrip(&target(TargetForm::World(worlds.worlds()[0].clone())));
        roundtrip(&target(TargetForm::Worlds(worlds.clone())));
        roundtrip(&target(TargetForm::Contrast(
            InterventionContrast::interaction(
                worlds,
                ScientificWorldId::from_raw(3),
                ScientificWorldId::from_raw(1),
                ScientificWorldId::from_raw(2),
                ScientificWorldId::from_raw(0),
            )
            .unwrap(),
        )));
        roundtrip(&target(TargetForm::Contrast(
            InterventionContrast::difference(
                WorldCollection::new([world(0, 0., 0., true), world(1, 4., 0., false)]).unwrap(),
                ScientificWorldId::from_raw(1),
                ScientificWorldId::from_raw(0),
                ScientificWorldId::from_raw(0),
            )
            .unwrap(),
        )));
    }
    #[test]
    fn scientific_target_wire_retains_temporal_history_and_mediation_premises() {
        let time = TimeFrame::temporal(
            3,
            HistoryWindow::max_lag(2),
            InitialLaw::named("retained pre-period group histories").unwrap(),
        )
        .unwrap();
        let action = WorldAction::schedule(
            Intervention::set(VariableId::from_raw(0), Value::Float64(4.)),
            TemporalPolicy::Sustained { from: 0, until: 2 },
        )
        .unwrap();
        let world = InterventionWorld::new(
            ScientificWorldId::from_raw(1),
            action,
            false,
            binding(),
            time.clone(),
        )
        .unwrap();
        roundtrip(&target(TargetForm::World(world)));
        let arms = MediationArms::new(
            VariableId::from_raw(0),
            [VariableId::from_raw(1)],
            Intervention::set(VariableId::from_raw(0), Value::Float64(0.)),
            Intervention::set(VariableId::from_raw(0), Value::Float64(4.)),
            MediationComponent::NaturalIndirect,
        )
        .unwrap();
        let mediation = MediationTarget::new(
            binding(),
            time,
            Some(TemporalPolicy::Pulse { at: -1 }),
            arms,
            [
                CrossWorldPremise::Composition,
                CrossWorldPremise::CrossWorldIndependence,
                CrossWorldPremise::PathSpecificNested,
            ],
        )
        .unwrap();
        roundtrip(&target(TargetForm::Mediation(mediation)));
    }
    #[test]
    fn scientific_target_wire_refuses_changed_question_and_future_version() {
        let mut wire =
            ScientificTargetWire::from_domain(&target(TargetForm::World(world(1, 4., 0., false))))
                .unwrap();
        if let FormWire::World { world } = &mut wire.target {
            world.binding.units = "EUR".into();
        }
        assert!(wire.to_domain().unwrap_err().to_string().contains("identity mismatch"));
        wire.version = SCIENTIFIC_TARGET_WIRE_VERSION + 1;
        assert!(matches!(wire.to_domain(), Err(IoError::UnsupportedVersion { .. })));
    }
    #[test]
    fn scientific_target_wire_revalidates_reference_and_premises_even_with_resealed_identity() {
        let arms = MediationArms::new(
            VariableId::from_raw(0),
            [VariableId::from_raw(1)],
            Intervention::set(VariableId::from_raw(0), Value::Float64(0.)),
            Intervention::set(VariableId::from_raw(0), Value::Float64(4.)),
            MediationComponent::NaturalDirect,
        )
        .unwrap();
        let mediation = MediationTarget::new(
            binding(),
            TimeFrame::static_decision(),
            None,
            arms,
            [
                CrossWorldPremise::Composition,
                CrossWorldPremise::CrossWorldIndependence,
                CrossWorldPremise::PathSpecificNested,
            ],
        )
        .unwrap();
        let mut wire =
            ScientificTargetWire::from_domain(&target(TargetForm::Mediation(mediation))).unwrap();
        if let FormWire::Mediation { premises, .. } = &mut wire.target {
            premises.clear();
        }
        assert!(
            wire.to_domain()
                .unwrap_err()
                .to_string()
                .contains("mediation requires explicit cross-world premises")
        );
        let mut value = serde_json::to_value(wire).unwrap();
        value["unrecognized_field"] = serde_json::json!(true);
        assert!(serde_json::from_value::<ScientificTargetWire>(value).is_err());
    }
    #[test]
    fn scientific_target_wire_retains_rectangular_dose_time_cells() {
        let mut worlds = Vec::new();
        let mut cells = Vec::new();
        for (dose_index, level) in [0.0, 4.0].into_iter().enumerate() {
            for (horizon_index, horizon) in [1, 3].into_iter().enumerate() {
                let id = ScientificWorldId::from_raw(
                    u32::try_from(dose_index * 2 + horizon_index).unwrap(),
                );
                let time = TimeFrame::temporal(
                    horizon,
                    HistoryWindow::max_lag(2),
                    InitialLaw::named("group histories").unwrap(),
                )
                .unwrap();
                let action = WorldAction::schedule(
                    Intervention::set(VariableId::from_raw(0), Value::Float64(level)),
                    TemporalPolicy::Pulse { at: 0 },
                )
                .unwrap();
                worlds.push(
                    InterventionWorld::new(id, action, id.raw() == 0, binding(), time).unwrap(),
                );
                cells.push(id);
            }
        }
        let surface = DoseTimeSurface::new(
            WorldCollection::new(worlds).unwrap(),
            VariableId::from_raw(0),
            [0., 4.],
            [1, 3],
            cells,
        )
        .unwrap();
        roundtrip(&target(TargetForm::DoseTime(surface)));
    }
    #[test]
    fn scientific_target_wire_roundtrips_general_weights_and_reads_previous_fixed_forms() {
        let worlds = WorldCollection::new([
            world(0, 0., 0., true),
            world(1, 4., 0., false),
            world(2, 0., 4., false),
        ])
        .unwrap();
        let contrast = InterventionContrast::linear(
            worlds,
            [
                (ScientificWorldId::from_raw(0), -0.5),
                (ScientificWorldId::from_raw(1), 0.25),
                (ScientificWorldId::from_raw(2), 0.25),
            ],
            ScientificWorldId::from_raw(0),
        )
        .unwrap();
        let scientific_target = target(TargetForm::Contrast(contrast));
        roundtrip(&scientific_target);
        let mut wire = ScientificTargetWire::from_domain(&scientific_target).unwrap();
        wire.version = 1;
        assert!(matches!(wire.to_domain(), Err(IoError::UnsupportedVersion { version: 1 })));
        let mut old =
            ScientificTargetWire::from_domain(&target(TargetForm::World(world(0, 0., 0., true))))
                .unwrap();
        old.version = 1;
        assert!(old.to_domain().is_ok());
    }
}
