// SPDX-License-Identifier: MIT OR Apache-2.0
//! Scientific problem, identification and contract payloads of a cyclic family.
//!
//! Everything here is derived from the typed request and its compiled solve
//! schedules alone, without fitting, sampling or solving, so a producer and an
//! independent consumer derive the same payloads. Keyed worlds become
//! scientific targets in the feedback vocabulary (equilibrium, continuous-time
//! and stationary frames; hard clamps, mechanism replacements, additive shocks
//! and initial-state changes). A specified or fitted model stays
//! model-conditional, a clamp-experiment recovery keeps its design-determined
//! unresolved directions, and a theorem-scoped cyclic identification keeps its
//! graphical source. None of these is upgraded to graphical identification of
//! an acyclic model.

use std::collections::BTreeSet;

use antecedent_core::{
    CanonicalWriter, Conditioning, GraphFamily, InferenceSelection, InitialLaw, InterventionWorld,
    ModelAssumptions, ObservationUnit, OutcomeQuantity, QuantityUnits, ScientificTarget,
    ScientificWorldId, StructuralAction, TargetForm, TargetPopulation, TimeFrame, VariableId,
    WorldAction, WorldBinding, WorldCollection,
};
use antecedent_io_core::bound_identity::CYCLIC_SOLUTION_CONTRACT;
use antecedent_io_core::contract_section::cyclic::{
    CyclicContractPayloads, CyclicDataSnapshotWire, CyclicIdentificationWire, CyclicInferenceWire,
    CyclicObservationWire, CyclicProgramWire, CyclicRecoveryWire, CyclicSolutionWire,
    CyclicTargetWire, CyclicUnresolvedWire,
};
use antecedent_io_core::scientific_target_wire::ScientificTargetWire;
use antecedent_model::compile::CompiledSystemPlan;
use serde::Serialize;

use super::{
    CyclicFamilyRequest as R, CyclicInterventionWire, CyclicWorldWire, ExperimentRegimeWire,
    RecoveredDrawOriginWire, RecoveryInferenceWire, ResamplingRuleWire, SamplingDesignWire,
    invalid,
};
use crate::cyclic_system_artifact::CyclicArtifactError;

/// What the uncertainty of a cyclic family's answer means.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CyclicUncertainty {
    /// A specified evaluation or a point fit: no uncertainty law is published.
    None,
    /// A sampling law (bootstrap, asymptotic normal, observed information).
    Sampling,
    /// A posterior or abduction law under a declared prior.
    Posterior,
}

/// How a cyclic family realizes its answer. Read by the lowering's statistical layer.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CyclicRealization {
    /// `specified_evaluation`, `point_fit`, `sampling_law`, `parameter_posterior`,
    /// `supplied_draws`, `exact_functional` or `graphical_reasoning`.
    pub realization: &'static str,
    /// Uncertainty method name; `none` when no law is produced.
    pub method: &'static str,
    /// What the published uncertainty means.
    pub uncertainty: CyclicUncertainty,
    /// Requested bootstrap or resampling replicates.
    pub replicates: Option<u64>,
    /// Requested posterior or law draws.
    pub draws: Option<u64>,
    /// Unit one replicate draws (`row`, `subject`, `cluster`, `calendar_block`).
    pub resample_unit: Option<&'static str>,
}

/// Layer one of a cyclic family.
#[derive(Clone, Debug, PartialEq)]
pub enum CyclicProblem {
    /// Keyed worlds with a declared outcome.
    Target(ScientificTarget),
    /// Keyed worlds answered for every model variable, one target per variable.
    JointTargets(Vec<ScientificTarget>),
    /// A functional that is not indexed by keyed worlds.
    Functional {
        /// `interventional_distribution`, `counterfactual` or `identified_set`.
        kind: &'static str,
        /// Outcome, when the functional reads one.
        outcome: Option<VariableId>,
        /// Digest of the query that defines the functional.
        query: [u8; 32],
    },
}

fn refuse(reason: impl std::fmt::Display) -> CyclicArtifactError {
    invalid(format!("cyclic contract: {reason}"))
}

fn json_digest<T: Serialize + ?Sized>(
    tag: &str,
    value: &T,
) -> Result<[u8; 32], CyclicArtifactError> {
    super::retained_recovery::json_digest(tag, value)
}

fn hex_bytes(hex: &str) -> Result<[u8; 32], CyclicArtifactError> {
    antecedent_core::digest_from_lower_hex(hex).ok_or_else(|| refuse("an identity is not a digest"))
}

/// Object fields of the encoded request selected by name, in name order.
///
/// The request is encoded once and read as a stream: unselected fields (the
/// data of a large experiment among them) are skipped without being built.
fn fields(request: &R, keys: &[&str]) -> Result<serde_json::Value, CyclicArtifactError> {
    if let R::RetainedExperimentalRecovery { inputs, .. } = request {
        let R::ExperimentalRecovery { options, inference, .. } = inputs.recipe() else {
            return Err(invalid("retained recovery recipe"));
        };
        let mut selected = std::collections::BTreeMap::new();
        if keys.contains(&"options") {
            selected.insert("options", serde_json::to_value(options).map_err(invalid)?);
        }
        if keys.contains(&"inference") {
            selected.insert("inference", serde_json::to_value(inference).map_err(invalid)?);
        }
        return serde_json::to_value(selected).map_err(invalid);
    }
    use serde::de::{DeserializeSeed, IgnoredAny, MapAccess, Visitor};

    struct Select<'k>(&'k [&'k str]);
    impl<'de> Visitor<'de> for Select<'_> {
        type Value = std::collections::BTreeMap<String, serde_json::Value>;
        fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter.write_str("an encoded cyclic family request")
        }
        fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
            let mut selected = std::collections::BTreeMap::new();
            while let Some(key) = map.next_key::<String>()? {
                if self.0.contains(&key.as_str()) {
                    let value: serde_json::Value = map.next_value()?;
                    selected.insert(key, value);
                } else {
                    map.next_value::<IgnoredAny>()?;
                }
            }
            Ok(selected)
        }
    }
    impl<'de> DeserializeSeed<'de> for Select<'_> {
        type Value = std::collections::BTreeMap<String, serde_json::Value>;
        fn deserialize<D: serde::Deserializer<'de>>(
            self,
            deserializer: D,
        ) -> Result<Self::Value, D::Error> {
            deserializer.deserialize_map(self)
        }
    }

    let encoded = serde_json::to_vec(request).map_err(invalid)?;
    let mut deserializer = serde_json::Deserializer::from_slice(&encoded);
    let selected = Select(keys).deserialize(&mut deserializer).map_err(invalid)?;
    Ok(serde_json::Value::Object(selected.into_iter().collect()))
}

/// The data a family reads, as a typed view: observations, rows, samples, laws or
/// supplied draws. A specified family reads none.
fn data_digest(request: &R) -> Result<[u8; 32], CyclicArtifactError> {
    let request = request.recovery_recipe();
    const TAG: &str = "antecedent.cyclic_family.data";
    match request {
        R::RetainedExperimentalRecovery { inputs, .. } => data_digest(inputs.recipe()),
        R::FrequentistBootstrap { rows, .. }
        | R::ParameterPosterior { rows, .. }
        | R::GradientFit { rows, .. }
        | R::OuFit { rows, .. } => json_digest(TAG, &("rows", rows)),
        R::JointCounterfactual { rows, observations, measurement_variances, .. } => {
            json_digest(TAG, &("rows_observations", rows, observations, measurement_variances))
        }
        R::PartialCounterfactual { observations, measurement_variances, .. } => {
            json_digest(TAG, &("observations", observations, measurement_variances))
        }
        R::AdamFit { training, validation, .. } => {
            json_digest(TAG, &("training_validation", training, validation))
        }
        R::StructureFit { training, heldout, .. } => {
            json_digest(TAG, &("training_heldout", training, heldout))
        }
        R::OdeFit { observations, .. } => json_digest(TAG, &("observations", observations)),
        R::PolynomialFit { measurements, .. } => json_digest(TAG, &("measurements", measurements)),
        R::OuMechanisms { regimes, .. } => json_digest(TAG, &("regimes", regimes)),
        R::Functional { law, .. } => json_digest(TAG, &("finite_law", law)),
        R::GaussianFunctional { law, .. } => json_digest(TAG, &("gaussian_law", law)),
        R::ExperimentalRecovery { samples, dropped, .. } => {
            json_digest(TAG, &("experiment_samples", samples, dropped))
        }
        R::RecoveredLinear { draws, .. } => json_digest(TAG, &("supplied_draws", draws)),
        R::DynamicInference { problem, .. } => json_digest(TAG, &("dynamic_problem", problem)),
        R::PolynomialCausal { program } => json_digest(TAG, &("polynomial_program", program)),
        R::System { request: system } => {
            json_digest(TAG, &("system", system.fit.as_ref().map(|fit| &fit.rows), &system.factual))
        }
        R::EquilibriumRoots { .. }
        | R::EquilibriumActions { .. }
        | R::Graph { .. }
        | R::Ode { .. }
        | R::OuStationary { .. }
        | R::OuTransient { .. }
        | R::Polynomial { .. }
        | R::LinearWorlds { .. }
        | R::EquilibriumWorlds { .. }
        | R::OdeWorlds { .. } => json_digest(TAG, &("specified",)),
    }
}

/// Fields that hold the specified model.
const MODEL_FIELDS: &[&str] =
    &["model", "parameters", "variables", "component", "nodes", "candidates", "program"];

/// Fields that hold the uncertainty method and its numerical settings.
const INFERENCE_FIELDS: &[&str] = &[
    "sampling",
    "prior_sd",
    "replicates",
    "seed",
    "design",
    "inference",
    "method",
    "uncertainty",
    "bootstrap_replicates",
    "threshold",
    "policy",
    "solve_policy",
    "options",
    "integration",
];

/// Fields that hold the solve, certificate and selection policy.
const SOLUTION_FIELDS: &[&str] = &[
    "certificate",
    "policy",
    "solve_policy",
    "initial",
    "initial_laws",
    "initial_mean",
    "initial_covariance",
    "schedule",
    "integration",
    "domain",
];

const fn unit_of_design(design: SamplingDesignWire) -> &'static str {
    match design {
        SamplingDesignWire::IndependentSubjectTrajectories => "subject",
        SamplingDesignWire::ClusteredPanel => "cluster",
        SamplingDesignWire::DependentTimeSeries => "calendar_block",
    }
}

const fn unit_of_rule(rule: ResamplingRuleWire) -> &'static str {
    match rule {
        ResamplingRuleWire::RowSubjectIid => "subject",
        ResamplingRuleWire::WholeCluster => "cluster",
        ResamplingRuleWire::ContiguousBlock => "calendar_block",
        ResamplingRuleWire::RowIid => "row",
    }
}

const fn realization(
    realization: &'static str,
    method: &'static str,
    uncertainty: CyclicUncertainty,
) -> CyclicRealization {
    CyclicRealization {
        realization,
        method,
        uncertainty,
        replicates: None,
        draws: None,
        resample_unit: None,
    }
}

const SPECIFIED: CyclicRealization =
    realization("specified_evaluation", "none", CyclicUncertainty::None);
const POINT_FIT: CyclicRealization = realization("point_fit", "none", CyclicUncertainty::None);

fn count(value: usize) -> Option<u64> {
    u64::try_from(value).ok()
}

/// How `request` realizes its answer.
#[must_use]
#[allow(clippy::too_many_lines, reason = "one exhaustive match over every cyclic family request")]
pub fn cyclic_realization(request: &R) -> CyclicRealization {
    let request = request.recovery_recipe();
    match request {
        R::RetainedExperimentalRecovery { inputs, .. } => cyclic_realization(inputs.recipe()),
        R::FrequentistBootstrap { replicates, .. } => CyclicRealization {
            replicates: count(*replicates),
            resample_unit: Some("row"),
            ..realization("sampling_law", "frequentist_bootstrap", CyclicUncertainty::Sampling)
        },
        R::ParameterPosterior { sampling, .. } => CyclicRealization {
            draws: count(sampling.chains.saturating_mul(sampling.draws)),
            ..realization(
                "parameter_posterior",
                "parameter_posterior",
                CyclicUncertainty::Posterior,
            )
        },
        R::JointCounterfactual { sampling, .. } => CyclicRealization {
            draws: count(sampling.chains.saturating_mul(sampling.draws)),
            ..realization(
                "parameter_posterior",
                "joint_abduction_posterior",
                CyclicUncertainty::Posterior,
            )
        },
        R::PartialCounterfactual { sampling, .. } => CyclicRealization {
            draws: count(sampling.chains.saturating_mul(sampling.draws)),
            ..realization(
                "parameter_posterior",
                "abduction_posterior",
                CyclicUncertainty::Posterior,
            )
        },
        R::StructureFit { bootstrap_replicates, .. } if *bootstrap_replicates > 0 => {
            CyclicRealization {
                replicates: count(*bootstrap_replicates),
                resample_unit: Some("row"),
                ..realization("point_fit", "selection_bootstrap", CyclicUncertainty::Sampling)
            }
        }
        R::StructureFit { .. }
        | R::AdamFit { .. }
        | R::GradientFit { .. }
        | R::PolynomialFit { .. }
        | R::OdeFit { .. }
        | R::OuFit { .. } => POINT_FIT,
        R::OuMechanisms { parameters, .. } if !parameters.is_empty() => POINT_FIT,
        R::System { request } if request.fit.is_some() => POINT_FIT,
        R::GaussianFunctional { .. } | R::Functional { .. } => {
            realization("exact_functional", "none", CyclicUncertainty::None)
        }
        R::PolynomialCausal { .. } => realization(
            "parameter_posterior",
            "finite_selection_posterior",
            CyclicUncertainty::Posterior,
        ),
        R::Graph { .. } => realization("graphical_reasoning", "none", CyclicUncertainty::None),
        R::RecoveredLinear { draws, uncertainty, .. } => match uncertainty {
            RecoveredDrawOriginWire::Supplied => CyclicRealization {
                draws: count(draws.len()),
                ..realization("supplied_draws", "supplied", CyclicUncertainty::None)
            },
            RecoveredDrawOriginWire::AsymptoticNormal { draws, .. } => CyclicRealization {
                draws: count(*draws),
                ..realization("sampling_law", "asymptotic_normal", CyclicUncertainty::Sampling)
            },
            RecoveredDrawOriginWire::Bootstrap { replicates, rule, .. } => CyclicRealization {
                replicates: count(*replicates),
                resample_unit: Some(unit_of_rule(*rule)),
                ..realization("sampling_law", "bootstrap", CyclicUncertainty::Sampling)
            },
            RecoveredDrawOriginWire::Posterior { draws, .. } => CyclicRealization {
                draws: count(*draws),
                ..realization("parameter_posterior", "posterior", CyclicUncertainty::Posterior)
            },
        },
        R::ExperimentalRecovery { inference, .. } => match inference {
            RecoveryInferenceWire::AsymptoticNormal { draws, .. } => CyclicRealization {
                draws: count(*draws),
                ..realization("sampling_law", "asymptotic_normal", CyclicUncertainty::Sampling)
            },
            RecoveryInferenceWire::Bootstrap { replicates, rule, .. } => CyclicRealization {
                replicates: count(*replicates),
                resample_unit: Some(unit_of_rule(*rule)),
                ..realization("sampling_law", "bootstrap", CyclicUncertainty::Sampling)
            },
            RecoveryInferenceWire::Posterior { draws, .. } => CyclicRealization {
                draws: count(*draws),
                ..realization("parameter_posterior", "posterior", CyclicUncertainty::Posterior)
            },
        },
        R::DynamicInference { method, .. } => match method {
            super::DynamicMethodWire::Bootstrap { replicates, design, .. } => CyclicRealization {
                replicates: count(*replicates),
                resample_unit: Some(unit_of_design(*design)),
                ..realization("sampling_law", "bootstrap", CyclicUncertainty::Sampling)
            },
            super::DynamicMethodWire::Information { .. } => {
                realization("sampling_law", "observed_information", CyclicUncertainty::Sampling)
            }
            super::DynamicMethodWire::Posterior { .. } => {
                realization("parameter_posterior", "posterior", CyclicUncertainty::Posterior)
            }
        },
        R::OuMechanisms { .. }
        | R::System { .. }
        | R::EquilibriumRoots { .. }
        | R::EquilibriumActions { .. }
        | R::Ode { .. }
        | R::OuStationary { .. }
        | R::OuTransient { .. }
        | R::Polynomial { .. }
        | R::LinearWorlds { .. }
        | R::EquilibriumWorlds { .. }
        | R::OdeWorlds { .. } => SPECIFIED,
    }
}

const fn inference_selection(realization: &CyclicRealization) -> InferenceSelection {
    match realization.uncertainty {
        CyclicUncertainty::None => InferenceSelection::Unspecified,
        CyclicUncertainty::Sampling => InferenceSelection::Frequentist,
        CyclicUncertainty::Posterior => InferenceSelection::Bayesian,
    }
}

fn structural_action(
    intervention: &CyclicInterventionWire,
) -> Result<StructuralAction, CyclicArtifactError> {
    match intervention {
        CyclicInterventionWire::Set { variable, value } => {
            StructuralAction::hard_clamp(VariableId::from_raw(*variable), *value)
        }
        CyclicInterventionWire::Shift { variable, delta } => {
            StructuralAction::additive_shock(VariableId::from_raw(*variable), *delta)
        }
        CyclicInterventionWire::Soft { variable, family, parameters } => {
            StructuralAction::replace_mechanism(
                VariableId::from_raw(*variable),
                family.as_str(),
                parameters,
            )
        }
    }
    .map_err(refuse)
}

/// State law names and their variables, for a continuous-time family.
type StateLaws<'a> = &'a [(String, u32)];

fn world_action(
    world: &CyclicWorldWire,
    laws: StateLaws<'_>,
) -> Result<WorldAction, CyclicArtifactError> {
    let mut actions =
        world.interventions.iter().map(structural_action).collect::<Result<Vec<_>, _>>()?;
    for state in &world.initial_states {
        let Some((_, variable)) = laws.iter().find(|(law, _)| *law == state.law) else {
            return Err(refuse(format!(
                "initial-state law `{}` names no declared state; next attempt: name a law the \
                 continuous-time family declares",
                state.law
            )));
        };
        actions.push(
            StructuralAction::initial_state(
                VariableId::from_raw(*variable),
                state.law.as_str(),
                state.level,
            )
            .map_err(refuse)?,
        );
    }
    WorldAction::structural(actions).map_err(refuse)
}

fn binding(outcome: u32, unit: &str) -> Result<WorldBinding, CyclicArtifactError> {
    WorldBinding::new(
        TargetPopulation::AllObserved,
        ObservationUnit::new(unit).map_err(refuse)?,
        OutcomeQuantity::mean(VariableId::from_raw(outcome), QuantityUnits::dimensionless())
            .map_err(refuse)?,
        Conditioning::unconditional(),
    )
    .map_err(refuse)
}

/// One keyed world before it is bound to an outcome.
struct KeyedWorld {
    id: u32,
    action: WorldAction,
}

fn worlds_target(
    worlds: &[KeyedWorld],
    outcome: u32,
    unit: &str,
    time: &TimeFrame,
    inference: InferenceSelection,
) -> Result<ScientificTarget, CyclicArtifactError> {
    let shared = binding(outcome, unit)?;
    let collection = WorldCollection::new(
        worlds
            .iter()
            .map(|world| {
                InterventionWorld::new(
                    ScientificWorldId::from_raw(world.id),
                    world.action.clone(),
                    false,
                    shared.clone(),
                    time.clone(),
                )
                .map_err(refuse)
            })
            .collect::<Result<Vec<_>, _>>()?,
    )
    .map_err(refuse)?;
    Ok(ScientificTarget::new(
        TargetForm::Worlds(collection),
        ModelAssumptions::new(GraphFamily::Cyclic, antecedent_core::AssumptionSet::new()),
        inference,
    ))
}

fn joint_targets(
    worlds: &[KeyedWorld],
    variables: &[u32],
    time: &TimeFrame,
    inference: InferenceSelection,
) -> Result<CyclicProblem, CyclicArtifactError> {
    if variables.is_empty() {
        return Err(refuse("a family with keyed worlds names at least one variable"));
    }
    Ok(CyclicProblem::JointTargets(
        variables
            .iter()
            .map(|variable| worlds_target(worlds, *variable, "system", time, inference))
            .collect::<Result<Vec<_>, _>>()?,
    ))
}

fn keyed(
    worlds: &[CyclicWorldWire],
    laws: StateLaws<'_>,
) -> Result<Vec<KeyedWorld>, CyclicArtifactError> {
    worlds
        .iter()
        .map(|world| Ok(KeyedWorld { id: world.world, action: world_action(world, laws)? }))
        .collect()
}

fn clamp_world(
    id: usize,
    clamps: impl IntoIterator<Item = (u32, f64)>,
) -> Result<KeyedWorld, CyclicArtifactError> {
    let action = WorldAction::structural(
        clamps
            .into_iter()
            .map(|(variable, level)| {
                StructuralAction::hard_clamp(VariableId::from_raw(variable), level)
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(refuse)?,
    )
    .map_err(refuse)?;
    Ok(KeyedWorld { id: u32::try_from(id).map_err(refuse)?, action })
}

fn position_id(position: usize) -> Result<u32, CyclicArtifactError> {
    u32::try_from(position).map_err(refuse)
}

fn functional(
    kind: &'static str,
    outcome: Option<VariableId>,
    request: &R,
) -> Result<CyclicProblem, CyclicArtifactError> {
    let mut writer = CanonicalWriter::new("antecedent.cyclic_family.functional_query");
    writer
        .text(request.family_name())
        .bytes(&json_digest("antecedent.cyclic_family.request.v1", request)?);
    Ok(CyclicProblem::Functional { kind, outcome, query: writer.finish() })
}

/// Layer one of `request`: keyed worlds in the feedback vocabulary, or the
/// functional the family answers.
///
/// Keyed worlds the vocabulary cannot state (none declared, or actions it
/// refuses) are lowered as the interventional functional the request defines,
/// never rewritten into other worlds.
///
/// # Errors
///
/// The request does not encode.
pub fn cyclic_problem(request: &R) -> Result<CyclicProblem, CyclicArtifactError> {
    match keyed_problem(request) {
        Ok(Some(problem)) => Ok(problem),
        Ok(None) | Err(_) => match request {
            R::JointCounterfactual { .. } | R::PartialCounterfactual { .. } => {
                functional("counterfactual", None, request)
            }
            R::EquilibriumRoots { .. } | R::Polynomial { .. } | R::PolynomialCausal { .. } => {
                functional("identified_set", None, request)
            }
            _ => functional("interventional_distribution", None, request),
        },
    }
}

/// Keyed worlds of `request` as scientific targets, or `None` for a family
/// without keyed worlds.
fn keyed_problem(request: &R) -> Result<Option<CyclicProblem>, CyclicArtifactError> {
    let inference = inference_selection(&cyclic_realization(request));
    let problem = match request {
        R::LinearWorlds { variables, worlds, .. }
        | R::EquilibriumWorlds { variables, worlds, .. } => {
            joint_targets(&keyed(worlds, &[])?, variables, &TimeFrame::equilibrium(), inference)
        }
        R::OdeWorlds { variables, initial_laws, times, worlds, .. } => {
            let laws: Vec<(String, u32)> =
                initial_laws.iter().cloned().zip(variables.iter().copied()).collect();
            let first = times.first().copied().unwrap_or(0.0);
            let last = times.last().copied().unwrap_or(0.0);
            let span = last - first;
            let horizon = if span > 0.0 { span } else { last };
            let law = InitialLaw::named(format!("initial_state:{}", initial_laws.join(",")))
                .map_err(refuse)?;
            let time = TimeFrame::continuous_time(horizon, law).map_err(refuse)?;
            joint_targets(&keyed(worlds, &laws)?, variables, &time, inference)
        }
        R::ExperimentalRecovery { worlds, outcome, .. }
        | R::RetainedExperimentalRecovery { worlds, outcome, .. } => {
            Ok(CyclicProblem::Target(worlds_target(
                &keyed(worlds, &[])?,
                *outcome,
                "experimental_unit",
                &TimeFrame::equilibrium(),
                inference,
            )?))
        }
        R::RecoveredLinear { nodes, worlds, outcome, .. } => {
            let node = |position: usize| {
                nodes.get(position).copied().ok_or_else(|| refuse("a clamp names no node"))
            };
            let keyed = worlds
                .iter()
                .enumerate()
                .map(|(index, world)| {
                    let clamps = world
                        .clamp_nodes
                        .iter()
                        .zip(&world.clamp_values)
                        .map(|(position, value)| Ok((node(*position)?, *value)))
                        .collect::<Result<Vec<_>, CyclicArtifactError>>()?;
                    clamp_world(index, clamps)
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(CyclicProblem::Target(worlds_target(
                &keyed,
                node(*outcome)?,
                "system",
                &TimeFrame::equilibrium(),
                inference,
            )?))
        }
        R::System { request: system } => {
            let keyed = system
                .worlds
                .iter()
                .enumerate()
                .map(|(index, clamps)| {
                    let clamps = clamps
                        .iter()
                        .map(|clamp| Ok((position_id(clamp.variable)?, clamp.value)))
                        .collect::<Result<Vec<_>, CyclicArtifactError>>()?;
                    clamp_world(index, clamps)
                })
                .collect::<Result<Vec<_>, _>>()?;
            let variables = (0..system.model.equations.len())
                .map(position_id)
                .collect::<Result<Vec<_>, _>>()?;
            joint_targets(&keyed, &variables, &TimeFrame::equilibrium(), inference)
        }
        _ => return Ok(None),
    };
    problem.map(Some)
}

/// Retained experimental units minus the mask.
fn retained(regimes: &[ExperimentRegimeWire], dropped: &[u32]) -> Vec<u32> {
    super::worlds::retained_units(regimes, dropped)
}

/// Directions a clamp design leaves unresolved before any fit: a coefficient
/// `K[target, source]` is covered when some regime clamps `source` while `target`
/// is passive in it.
#[must_use]
pub fn design_unresolved(
    regimes: &[ExperimentRegimeWire],
    dropped: &[u32],
) -> Vec<CyclicUnresolvedWire> {
    let nodes = retained(regimes, dropped);
    let dropped: BTreeSet<u32> = dropped.iter().copied().collect();
    let designs: Vec<(BTreeSet<u32>, BTreeSet<u32>)> = regimes
        .iter()
        .map(|regime| {
            let clamped: BTreeSet<u32> = regime.clamps.iter().map(|clamp| clamp.unit).collect();
            let passive = regime
                .units
                .iter()
                .copied()
                .filter(|unit| !clamped.contains(unit) && !dropped.contains(unit))
                .collect();
            (clamped, passive)
        })
        .collect();
    nodes
        .iter()
        .filter_map(|target| {
            let missing: Vec<u32> = nodes
                .iter()
                .copied()
                .filter(|source| {
                    source != target
                        && !designs.iter().any(|(clamped, passive)| {
                            clamped.contains(source) && passive.contains(target)
                        })
                })
                .collect();
            (!missing.is_empty())
                .then_some(CyclicUnresolvedWire { target: *target, missing_clamps: missing })
        })
        .collect()
}

/// Layer two of `request`. None of these is a graphical identification product
/// of an acyclic model.
///
/// # Errors
///
/// A request field that does not encode.
pub fn cyclic_identification(request: &R) -> Result<CyclicIdentificationWire, CyclicArtifactError> {
    let request = request.recovery_recipe();
    Ok(match request {
        R::ExperimentalRecovery { regimes, dropped, .. } => {
            CyclicIdentificationWire::ExperimentalRecovery {
                design: json_digest("antecedent.cyclic_family.clamp_design", &(regimes, dropped))?,
                unresolved: design_unresolved(regimes, dropped),
            }
        }
        R::Graph { request: graph } => CyclicIdentificationWire::CyclicIoScm {
            source: json_digest("antecedent.cyclic_family.ioscm_source", graph)?,
        },
        R::Functional { source, source_identity, .. }
        | R::GaussianFunctional { source, source_identity, .. } => {
            CyclicIdentificationWire::CyclicIoScm {
                source: json_digest(
                    "antecedent.cyclic_family.ioscm_source",
                    &(source, source_identity),
                )?,
            }
        }
        R::System { request: system } => CyclicIdentificationWire::SpecifiedModelConditional {
            model: json_digest(
                "antecedent.cyclic_family.model",
                &(&system.model, system.fit.as_ref().map(|fit| &fit.parameters)),
            )?,
        },
        other => CyclicIdentificationWire::SpecifiedModelConditional {
            model: json_digest("antecedent.cyclic_family.model", &fields(other, MODEL_FIELDS)?)?,
        },
    })
}

fn modality(request: &R) -> &'static str {
    let request = request.recovery_recipe();
    match request {
        R::ExperimentalRecovery { .. } => "clamp_experiments",
        R::RecoveredLinear { .. } => "supplied_draws",
        R::Functional { .. } | R::GaussianFunctional { .. } => "finite_law",
        R::OdeFit { .. } | R::OuFit { .. } | R::DynamicInference { .. } => "trajectories",
        R::OuMechanisms { regimes, .. } if !regimes.is_empty() => "stationary_regimes",
        R::System { request } if request.fit.is_some() => "equilibrium_rows",
        R::AdamFit { .. }
        | R::GradientFit { .. }
        | R::FrequentistBootstrap { .. }
        | R::ParameterPosterior { .. }
        | R::JointCounterfactual { .. }
        | R::PartialCounterfactual { .. }
        | R::StructureFit { .. }
        | R::PolynomialFit { .. }
        | R::PolynomialCausal { .. } => "equilibrium_rows",
        _ => "specified",
    }
}

fn variables(request: &R) -> Vec<u32> {
    let request = request.recovery_recipe();
    match request {
        R::LinearWorlds { variables, .. }
        | R::EquilibriumWorlds { variables, .. }
        | R::OdeWorlds { variables, .. } => variables.clone(),
        R::RecoveredLinear { nodes, .. } => nodes.clone(),
        R::ExperimentalRecovery { regimes, dropped, .. } => retained(regimes, dropped),
        R::System { request } => (0..request.model.equations.len())
            .filter_map(|index| u32::try_from(index).ok())
            .collect(),
        _ => Vec::new(),
    }
}

fn solution_kind(request: &R) -> &'static str {
    let request = request.recovery_recipe();
    match request {
        R::LinearWorlds { .. } => "linear_family_certificate",
        R::RecoveredLinear { .. } | R::ExperimentalRecovery { .. } => "hard_clamp_worlds",
        R::Ode { .. } | R::OdeWorlds { .. } | R::OdeFit { .. } => "initial_value",
        R::OuStationary { .. } | R::OuFit { .. } | R::OuMechanisms { .. } => "stationary_law",
        R::OuTransient { .. } => "stochastic_transient",
        R::EquilibriumRoots { .. }
        | R::Polynomial { .. }
        | R::PolynomialFit { .. }
        | R::PolynomialCausal { .. } => "isolated_solution_set",
        R::Graph { .. } | R::Functional { .. } | R::GaussianFunctional { .. } => "theorem_scoped",
        R::DynamicInference { .. } => "dynamic_family",
        _ => "equilibrium_solve",
    }
}

/// Every contract payload of `request`, derived from the request and its
/// compiled solve schedules without fitting or solving.
///
/// # Errors
///
/// A keyed world the vocabulary refuses, or a field that does not encode.
pub fn cyclic_contract_payloads(
    request: &R,
    plans: &[CompiledSystemPlan],
    fit_dependencies: &str,
) -> Result<CyclicContractPayloads, CyclicArtifactError> {
    let family = request.family_name().to_owned();
    let realization = cyclic_realization(request);
    let target = match cyclic_problem(request)? {
        CyclicProblem::Target(target) => CyclicTargetWire::Worlds {
            target: Box::new(ScientificTargetWire::from_domain(&target).map_err(invalid)?),
        },
        CyclicProblem::JointTargets(targets) => CyclicTargetWire::JointWorlds {
            targets: targets
                .iter()
                .map(|target| ScientificTargetWire::from_domain(target).map_err(invalid))
                .collect::<Result<Vec<_>, _>>()?,
        },
        CyclicProblem::Functional { kind, outcome, query } => CyclicTargetWire::Functional {
            kind: kind.to_owned(),
            outcome: outcome.map(VariableId::raw),
            query,
        },
    };
    let units = match request.recovery_recipe() {
        R::ExperimentalRecovery { regimes, dropped, .. } => retained(regimes, dropped),
        _ => Vec::new(),
    };
    let recovery = match request.recovery_recipe() {
        R::ExperimentalRecovery { regimes, dropped, samples, options, inference, .. } => {
            Some(CyclicRecoveryWire {
                experiment_data: json_digest("antecedent.cyclic_family.experiment_data", samples)?,
                regime_contract: json_digest(
                    "antecedent.cyclic_family.regime_contract",
                    &(regimes, dropped),
                )?,
                derivation: json_digest(
                    "antecedent.cyclic_family.recovery_derivation",
                    &(
                        antecedent_identify::experimental_recovery::CLAMP_EXPERIMENT_RECIPE,
                        options,
                        inference,
                        design_unresolved(regimes, dropped),
                    ),
                )?,
            })
        }
        _ => None,
    };
    let content = data_digest(request)?;
    let settings = match request {
        R::System { request: system } => json_digest(
            "antecedent.cyclic_family.inference",
            &(system.fit.as_ref().map(|fit| &fit.options), &system.policy),
        )?,
        other => {
            json_digest("antecedent.cyclic_family.inference", &fields(other, INFERENCE_FIELDS)?)?
        }
    };
    let solution_settings = match request {
        R::System { request: system } => json_digest(
            "antecedent.cyclic_family.solution_policy",
            &(&system.policy, &system.initial),
        )?,
        other => json_digest(
            "antecedent.cyclic_family.solution_policy",
            &fields(other, SOLUTION_FIELDS)?,
        )?,
    };
    Ok(CyclicContractPayloads {
        family: family.clone(),
        target,
        identification: cyclic_identification(request)?,
        program: CyclicProgramWire {
            family: family.clone(),
            realization: realization.realization.to_owned(),
            fit_dependencies: hex_bytes(fit_dependencies)?,
        },
        inference_binding: CyclicInferenceWire { method: realization.method.to_owned(), settings },
        observation: CyclicObservationWire {
            modality: modality(request).to_owned(),
            variables: variables(request),
            units,
        },
        data_snapshot: CyclicDataSnapshotWire { content, recovery },
        solution: CyclicSolutionWire {
            contract: CYCLIC_SOLUTION_CONTRACT.to_owned(),
            family,
            schedules: json_digest("antecedent.cyclic_family.solve_schedules", plans)?,
            policy: format!(
                "{}:{}",
                solution_kind(request),
                antecedent_core::to_hex(&solution_settings)
            ),
        },
    })
}
