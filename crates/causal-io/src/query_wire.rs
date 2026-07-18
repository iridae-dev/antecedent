//! Full CausalQuery and Intervention wire forms (DESIGN.md §24).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use causal_core::{
    AllocationMethod, AnomalyAttributionQuery, AttributionComponents, AverageEffectQuery,
    CausalQuery, ChangeAttributionQuery, ConditionalEffectQuery, CounterfactualQuery,
    DistributionRef, DynamicRuleId, EnvironmentId, Intervention, InterventionSequence,
    InterventionalDistributionQuery, MechanismChangeQuery, MechanismOverride, MediationContrast,
    MediationQuery, OrderedFloatBits, PathSpecificEffectQuery, PopulationSelector, PredicateExpr,
    SequencedIntervention, ShapleyConfig, ShapleyMode, StochasticPolicy, TargetPopulation,
    TemporalEffectQuery, TemporalPolicy, UnitChangeQuery, Value, VariableId,
};
use serde::{Deserialize, Serialize};

use crate::convert::{vars_from_raw, vars_to_raw};
use crate::error::IoError;

/// Wire scalar value.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ValueWire {
    /// Float64.
    Float64(f64),
    /// Int64.
    Int64(i64),
    /// Bool.
    Bool(bool),
    /// Category code.
    Category(u32),
    /// Diagnostic label.
    Label(String),
}

impl ValueWire {
    /// Encode.
    #[must_use]
    pub fn from_value(v: &Value) -> Self {
        match v {
            Value::Float64(x) => Self::Float64(*x),
            Value::Int64(x) => Self::Int64(*x),
            Value::Bool(x) => Self::Bool(*x),
            Value::Category(x) => Self::Category(*x),
            Value::Label(s) => Self::Label(s.to_string()),
        }
    }

    /// Decode.
    #[must_use]
    pub fn to_value(&self) -> Value {
        match self {
            Self::Float64(x) => Value::Float64(*x),
            Self::Int64(x) => Value::Int64(*x),
            Self::Bool(x) => Value::Bool(*x),
            Self::Category(x) => Value::Category(*x),
            Self::Label(s) => Value::Label(Arc::from(s.as_str())),
        }
    }
}

/// Hard set intervention on the wire (kept for posterior/distribution helpers).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SetInterventionWire {
    /// Target variable raw id.
    pub variable: u32,
    /// Assigned value.
    pub value: ValueWire,
}

/// Target population on the wire.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TargetPopulationWire {
    /// All observed units.
    AllObserved,
    /// Treated units.
    Treated,
    /// Untreated units.
    Untreated,
    /// Environment-restricted.
    Environment(u32),
    /// Named registry predicate.
    PredicateNamed(String),
    /// Explicit row indices.
    PredicateRows(Vec<u64>),
    /// Custom distribution handle.
    CustomDistribution(u32),
}

impl TargetPopulationWire {
    /// Encode.
    ///
    /// # Errors
    ///
    /// Unknown variants or row indices that do not fit `u64`.
    pub fn from_domain(p: &TargetPopulation) -> Result<Self, IoError> {
        Ok(match p {
            TargetPopulation::AllObserved => Self::AllObserved,
            TargetPopulation::Treated => Self::Treated,
            TargetPopulation::Untreated => Self::Untreated,
            TargetPopulation::Environment(id) => Self::Environment(id.raw()),
            TargetPopulation::Predicate(PredicateExpr::Named(name)) => {
                Self::PredicateNamed(name.to_string())
            }
            TargetPopulation::Predicate(PredicateExpr::Rows(rows)) => Self::PredicateRows(
                rows.iter()
                    .map(|&r| u64::try_from(r).map_err(|_| IoError::TooLarge))
                    .collect::<Result<Vec<_>, _>>()?,
            ),
            TargetPopulation::Predicate(other) => {
                return Err(IoError::Convert(format!(
                    "unsupported PredicateExpr for query wire: {other:?}"
                )));
            }
            TargetPopulation::CustomDistribution(r) => Self::CustomDistribution(r.raw()),
            other => {
                return Err(IoError::Convert(format!(
                    "unsupported TargetPopulation for query wire: {other:?}"
                )));
            }
        })
    }

    /// Decode.
    ///
    /// # Errors
    ///
    /// Row indices that do not fit `usize`.
    pub fn to_domain(&self) -> Result<TargetPopulation, IoError> {
        Ok(match self {
            Self::AllObserved => TargetPopulation::AllObserved,
            Self::Treated => TargetPopulation::Treated,
            Self::Untreated => TargetPopulation::Untreated,
            Self::Environment(raw) => TargetPopulation::Environment(EnvironmentId::from_raw(*raw)),
            Self::PredicateNamed(name) => {
                TargetPopulation::Predicate(PredicateExpr::named(name.as_str()))
            }
            Self::PredicateRows(rows) => {
                let idxs = rows
                    .iter()
                    .map(|&r| usize::try_from(r).map_err(|_| IoError::TooLarge))
                    .collect::<Result<Vec<_>, _>>()?;
                TargetPopulation::Predicate(PredicateExpr::rows(idxs))
            }
            Self::CustomDistribution(raw) => {
                TargetPopulation::CustomDistribution(DistributionRef::from_raw(*raw))
            }
        })
    }
}

/// Temporal policy wire.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TemporalPolicyWire {
    /// Pulse.
    Pulse {
        /// Offset.
        at: i32,
    },
    /// Sustained inclusive range.
    Sustained {
        /// From.
        from: i32,
        /// Until.
        until: i32,
    },
    /// Dynamic rule handle.
    Dynamic {
        /// Rule id.
        rule: u32,
    },
}

impl TemporalPolicyWire {
    fn from_domain(p: &TemporalPolicy) -> Result<Self, IoError> {
        Ok(match p {
            TemporalPolicy::Pulse { at } => Self::Pulse { at: *at },
            TemporalPolicy::Sustained { from, until } => {
                Self::Sustained { from: *from, until: *until }
            }
            TemporalPolicy::Dynamic { rule } => Self::Dynamic { rule: rule.raw() },
            other => {
                return Err(IoError::Convert(format!("unsupported TemporalPolicy: {other:?}")));
            }
        })
    }

    #[must_use]
    fn to_domain(&self) -> TemporalPolicy {
        match self {
            Self::Pulse { at } => TemporalPolicy::Pulse { at: *at },
            Self::Sustained { from, until } => TemporalPolicy::Sustained { from: *from, until: *until },
            Self::Dynamic { rule } => TemporalPolicy::Dynamic {
                rule: DynamicRuleId::from_raw(*rule),
            },
        }
    }
}

/// Stochastic policy wire.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum StochasticPolicyWire {
    /// Bernoulli.
    Bernoulli {
        /// p.
        p: f64,
    },
    /// Gaussian.
    Gaussian {
        /// Mean.
        mean: f64,
        /// Variance.
        variance: f64,
    },
    /// Categorical.
    Categorical {
        /// Probabilities.
        probs: Vec<f64>,
    },
}

impl StochasticPolicyWire {
    fn from_domain(p: &StochasticPolicy) -> Result<Self, IoError> {
        Ok(match p {
            StochasticPolicy::Bernoulli { p } => Self::Bernoulli { p: *p },
            StochasticPolicy::Gaussian { mean, variance } => {
                Self::Gaussian { mean: *mean, variance: *variance }
            }
            StochasticPolicy::Categorical { probs } => Self::Categorical { probs: probs.to_vec() },
            other => {
                return Err(IoError::Convert(format!("unsupported StochasticPolicy: {other:?}")));
            }
        })
    }

    #[must_use]
    fn to_domain(&self) -> StochasticPolicy {
        match self {
            Self::Bernoulli { p } => StochasticPolicy::Bernoulli { p: *p },
            Self::Gaussian { mean, variance } => {
                StochasticPolicy::Gaussian { mean: *mean, variance: *variance }
            }
            Self::Categorical { probs } => {
                StochasticPolicy::Categorical { probs: Arc::from(probs.as_slice()) }
            }
        }
    }
}

/// Soft mechanism override wire.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct MechanismOverrideWire {
    /// Family id.
    pub family_id: String,
    /// Parameters.
    pub parameters: Vec<f64>,
}

/// Full intervention wire.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum InterventionWire {
    /// Hard set.
    Set {
        /// Variable.
        variable: u32,
        /// Value.
        value: ValueWire,
    },
    /// Shift.
    Shift {
        /// Variable.
        variable: u32,
        /// Delta.
        delta: ValueWire,
    },
    /// Stochastic.
    Stochastic {
        /// Variable.
        variable: u32,
        /// Policy.
        policy: StochasticPolicyWire,
    },
    /// Soft.
    Soft {
        /// Variable.
        variable: u32,
        /// Mechanism.
        mechanism: MechanismOverrideWire,
    },
    /// Sequence.
    Sequence {
        /// Steps.
        steps: Vec<SequencedInterventionWire>,
    },
}

/// Sequenced intervention step.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SequencedInterventionWire {
    /// Nested intervention.
    pub intervention: Box<InterventionWire>,
    /// Temporal policy.
    pub temporal: TemporalPolicyWire,
}

impl InterventionWire {
    /// Encode.
    ///
    /// # Errors
    ///
    /// Unknown intervention variants.
    pub fn from_domain(iv: &Intervention) -> Result<Self, IoError> {
        Ok(match iv {
            Intervention::Set { variable, value } => Self::Set {
                variable: variable.raw(),
                value: ValueWire::from_value(value),
            },
            Intervention::Shift { variable, delta } => Self::Shift {
                variable: variable.raw(),
                delta: ValueWire::from_value(delta),
            },
            Intervention::Stochastic { variable, policy } => Self::Stochastic {
                variable: variable.raw(),
                policy: StochasticPolicyWire::from_domain(policy)?,
            },
            Intervention::Soft { variable, mechanism } => Self::Soft {
                variable: variable.raw(),
                mechanism: MechanismOverrideWire {
                    family_id: mechanism.family_id.to_string(),
                    parameters: mechanism.parameters.to_vec(),
                },
            },
            Intervention::Sequence(seq) => Self::Sequence {
                steps: seq
                    .steps
                    .iter()
                    .map(|s| {
                        Ok(SequencedInterventionWire {
                            intervention: Box::new(Self::from_domain(&s.intervention)?),
                            temporal: TemporalPolicyWire::from_domain(&s.temporal)?,
                        })
                    })
                    .collect::<Result<Vec<_>, IoError>>()?,
            },
            other => {
                return Err(IoError::Convert(format!("unsupported Intervention: {other:?}")));
            }
        })
    }

    /// Decode.
    #[must_use]
    pub fn to_domain(&self) -> Intervention {
        match self {
            Self::Set { variable, value } => {
                Intervention::set(VariableId::from_raw(*variable), value.to_value())
            }
            Self::Shift { variable, delta } => {
                Intervention::shift(VariableId::from_raw(*variable), delta.to_value())
            }
            Self::Stochastic { variable, policy } => Intervention::stochastic(
                VariableId::from_raw(*variable),
                policy.to_domain(),
            ),
            Self::Soft { variable, mechanism } => Intervention::soft(
                VariableId::from_raw(*variable),
                MechanismOverride {
                    family_id: Arc::from(mechanism.family_id.as_str()),
                    parameters: Arc::from(mechanism.parameters.as_slice()),
                },
            ),
            Self::Sequence { steps } => Intervention::sequence(InterventionSequence::new(
                steps
                    .iter()
                    .map(|s| SequencedIntervention {
                        intervention: s.intervention.to_domain(),
                        temporal: s.temporal.to_domain(),
                    })
                    .collect::<Vec<_>>(),
            )),
        }
    }
}

/// Wire form of [`InterventionalDistributionQuery`].
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct InterventionalDistributionQueryWire {
    /// Outcome variable raw ids.
    pub outcomes: Vec<u32>,
    /// Interventions (full).
    pub interventions: Vec<InterventionWire>,
    /// Target population.
    pub target_population: TargetPopulationWire,
}

/// Wire form of [`PathSpecificEffectQuery`].
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct PathSpecificEffectQueryWire {
    /// Treatment raw id.
    pub treatment: u32,
    /// Outcome raw id.
    pub outcome: u32,
    /// Intermediate path-node raw ids.
    pub path_nodes: Vec<u32>,
    /// Control.
    pub control: InterventionWire,
    /// Active.
    pub active: InterventionWire,
    /// Target population.
    pub target_population: TargetPopulationWire,
    /// Max paths.
    pub max_paths: u64,
    /// Max path length.
    pub max_len: u64,
}

/// Population selector wire.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PopulationSelectorWire {
    /// All.
    All,
    /// Explicit rows.
    Rows(Vec<u64>),
    /// Environment index.
    Environment {
        /// Index.
        env_index: u64,
    },
    /// Time range.
    TimeRange {
        /// Start.
        start: u64,
        /// End.
        end: u64,
    },
}

/// Full causal query wire.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum CausalQueryWire {
    /// Average effect.
    AverageEffect {
        /// Treatment.
        treatment: u32,
        /// Outcome.
        outcome: u32,
        /// Modifiers.
        effect_modifiers: Vec<u32>,
        /// Control.
        control: InterventionWire,
        /// Active.
        active: InterventionWire,
        /// Population.
        target_population: TargetPopulationWire,
    },
    /// Temporal effect.
    TemporalEffect {
        /// Treatment.
        treatment: u32,
        /// Outcome.
        outcome: u32,
        /// Policy.
        policy: TemporalPolicyWire,
        /// Control.
        control: InterventionWire,
        /// Active.
        active: InterventionWire,
        /// Horizon.
        horizon_steps: u32,
        /// Max history lag.
        max_history_lag: Option<u32>,
        /// Population.
        target_population: TargetPopulationWire,
    },
    /// Counterfactual.
    Counterfactual {
        /// Outcomes.
        outcomes: Vec<u32>,
        /// Interventions.
        interventions: Vec<InterventionWire>,
        /// Nested flag.
        allow_nested: bool,
    },
    /// Anomaly attribution.
    AnomalyAttribution {
        /// Targets.
        targets: Vec<u32>,
        /// Optional rows.
        unit_rows: Option<Vec<u64>>,
        /// Cap.
        max_units: u64,
    },
    /// Change attribution.
    ChangeAttribution {
        /// Outcome.
        outcome: u32,
        /// Baseline.
        baseline: PopulationSelectorWire,
        /// Comparison.
        comparison: PopulationSelectorWire,
        /// Components.
        components: String,
        /// Allocation.
        allocation: AllocationMethodWire,
        /// Cap.
        max_components: u64,
    },
    /// Mechanism change.
    MechanismChange {
        /// Targets.
        targets: Vec<u32>,
        /// Baseline.
        baseline: PopulationSelectorWire,
        /// Comparison.
        comparison: PopulationSelectorWire,
        /// Alpha bits.
        significance_level_bits: u64,
        /// Cap.
        max_targets: u64,
    },
    /// Unit change.
    UnitChange {
        /// Outcome.
        outcome: u32,
        /// Rows.
        unit_rows: Option<Vec<u64>>,
        /// Components.
        components: String,
        /// Allocation.
        allocation: AllocationMethodWire,
        /// Cap.
        max_units: u64,
    },
    /// Mediation.
    Mediation {
        /// Treatment.
        treatment: u32,
        /// Outcome.
        outcome: u32,
        /// Mediators.
        mediators: Vec<u32>,
        /// Contrast.
        contrast: String,
        /// Control.
        control: InterventionWire,
        /// Active.
        active: InterventionWire,
        /// Population.
        target_population: TargetPopulationWire,
    },
    /// Conditional effect.
    ConditionalEffect {
        /// Inner average-effect query.
        inner: Box<CausalQueryWire>,
    },
    /// Interventional distribution.
    Distribution(InterventionalDistributionQueryWire),
    /// Path-specific.
    PathSpecific(PathSpecificEffectQueryWire),
}

/// Allocation method wire.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AllocationMethodWire {
    /// Sequential.
    Sequential {
        /// Component raw ids.
        order: Vec<u32>,
    },
    /// Shapley.
    Shapley {
        /// Mode tag.
        mode: String,
        /// Mode parameter (samples / permutations).
        n: u64,
        /// Exact component cap.
        max_exact_components: u64,
        /// Override flag.
        allow_exact_override: bool,
        /// Seed.
        seed: u64,
    },
    /// Path-based.
    PathBased,
}

fn population_to_wire(p: &PopulationSelector) -> Result<PopulationSelectorWire, IoError> {
    Ok(match p {
        PopulationSelector::All => PopulationSelectorWire::All,
        PopulationSelector::Rows(rows) => PopulationSelectorWire::Rows(
            rows.iter()
                .map(|&r| u64::try_from(r).map_err(|_| IoError::TooLarge))
                .collect::<Result<Vec<_>, _>>()?,
        ),
        PopulationSelector::Environment { env_index } => PopulationSelectorWire::Environment {
            env_index: u64::try_from(*env_index).map_err(|_| IoError::TooLarge)?,
        },
        PopulationSelector::TimeRange { start, end } => PopulationSelectorWire::TimeRange {
            start: u64::try_from(*start).map_err(|_| IoError::TooLarge)?,
            end: u64::try_from(*end).map_err(|_| IoError::TooLarge)?,
        },
        other => {
            return Err(IoError::Convert(format!("unsupported PopulationSelector: {other:?}")));
        }
    })
}

fn population_from_wire(p: &PopulationSelectorWire) -> Result<PopulationSelector, IoError> {
    Ok(match p {
        PopulationSelectorWire::All => PopulationSelector::All,
        PopulationSelectorWire::Rows(rows) => PopulationSelector::Rows(
            rows.iter()
                .map(|&r| usize::try_from(r).map_err(|_| IoError::TooLarge))
                .collect::<Result<Vec<_>, _>>()?
                .into(),
        ),
        PopulationSelectorWire::Environment { env_index } => PopulationSelector::Environment {
            env_index: usize::try_from(*env_index).map_err(|_| IoError::TooLarge)?,
        },
        PopulationSelectorWire::TimeRange { start, end } => PopulationSelector::TimeRange {
            start: usize::try_from(*start).map_err(|_| IoError::TooLarge)?,
            end: usize::try_from(*end).map_err(|_| IoError::TooLarge)?,
        },
    })
}

fn components_to_str(c: AttributionComponents) -> Result<&'static str, IoError> {
    Ok(match c {
        AttributionComponents::Inputs => "inputs",
        AttributionComponents::Mechanisms => "mechanisms",
        AttributionComponents::Structure => "structure",
        AttributionComponents::InputsAndMechanisms => "inputs_and_mechanisms",
        AttributionComponents::All => "all",
        _ => return Err(IoError::Convert("unsupported AttributionComponents".into())),
    })
}

fn components_from_str(s: &str) -> Result<AttributionComponents, IoError> {
    Ok(match s {
        "inputs" => AttributionComponents::Inputs,
        "mechanisms" => AttributionComponents::Mechanisms,
        "structure" => AttributionComponents::Structure,
        "inputs_and_mechanisms" => AttributionComponents::InputsAndMechanisms,
        "all" => AttributionComponents::All,
        other => {
            return Err(IoError::Convert(format!("unknown AttributionComponents `{other}`")));
        }
    })
}

fn mediation_contrast_to_str(c: MediationContrast) -> Result<&'static str, IoError> {
    Ok(match c {
        MediationContrast::Total => "total",
        MediationContrast::Direct => "direct",
        MediationContrast::Mediated => "mediated",
        MediationContrast::NaturalDirect => "natural_direct",
        MediationContrast::NaturalIndirect => "natural_indirect",
    })
}

fn mediation_contrast_from_str(s: &str) -> Result<MediationContrast, IoError> {
    Ok(match s {
        "total" => MediationContrast::Total,
        "direct" => MediationContrast::Direct,
        "mediated" => MediationContrast::Mediated,
        "natural_direct" => MediationContrast::NaturalDirect,
        "natural_indirect" => MediationContrast::NaturalIndirect,
        other => return Err(IoError::Convert(format!("unknown MediationContrast `{other}`"))),
    })
}

fn allocation_to_wire(a: &AllocationMethod) -> Result<AllocationMethodWire, IoError> {
    Ok(match a {
        AllocationMethod::Sequential { order } => AllocationMethodWire::Sequential {
            order: order.iter().map(|c| c.raw()).collect(),
        },
        AllocationMethod::PathBased => AllocationMethodWire::PathBased,
        AllocationMethod::Shapley { approximation } => {
            let (mode, n) = match approximation.mode {
                ShapleyMode::Exact => ("exact", 0u64),
                ShapleyMode::MonteCarlo { n_samples } => {
                    ("monte_carlo", u64::try_from(n_samples).unwrap_or(u64::MAX))
                }
                ShapleyMode::Permutation { n_permutations } => {
                    ("permutation", u64::try_from(n_permutations).unwrap_or(u64::MAX))
                }
                _ => {
                    return Err(IoError::Convert("unsupported ShapleyMode".into()));
                }
            };
            AllocationMethodWire::Shapley {
                mode: mode.into(),
                n,
                max_exact_components: u64::try_from(approximation.max_exact_components)
                    .unwrap_or(u64::MAX),
                allow_exact_override: approximation.allow_exact_override,
                seed: approximation.seed,
            }
        }
        _ => return Err(IoError::Convert("unsupported AllocationMethod".into())),
    })
}

fn allocation_from_wire(a: &AllocationMethodWire) -> Result<AllocationMethod, IoError> {
    Ok(match a {
        AllocationMethodWire::Sequential { order } => AllocationMethod::Sequential {
            order: order
                .iter()
                .copied()
                .map(causal_core::ComponentId::from_raw)
                .collect::<Vec<_>>()
                .into(),
        },
        AllocationMethodWire::PathBased => AllocationMethod::PathBased,
        AllocationMethodWire::Shapley {
            mode,
            n,
            max_exact_components,
            allow_exact_override,
            seed,
        } => {
            let n_usize = usize::try_from(*n).map_err(|_| IoError::TooLarge)?;
            let mode = match mode.as_str() {
                "exact" => ShapleyMode::Exact,
                "monte_carlo" => ShapleyMode::MonteCarlo { n_samples: n_usize },
                "permutation" => ShapleyMode::Permutation { n_permutations: n_usize },
                other => {
                    return Err(IoError::Convert(format!("unknown ShapleyMode `{other}`")));
                }
            };
            AllocationMethod::Shapley {
                approximation: ShapleyConfig {
                    mode,
                    max_exact_components: usize::try_from(*max_exact_components)
                        .map_err(|_| IoError::TooLarge)?,
                    allow_exact_override: *allow_exact_override,
                    seed: *seed,
                },
            }
        }
    })
}

/// Encode any [`CausalQuery`].
///
/// # Errors
///
/// Unsupported nested fields.
pub fn causal_query_to_wire(q: &CausalQuery) -> Result<CausalQueryWire, IoError> {
    Ok(match q {
        CausalQuery::AverageEffect(q) => CausalQueryWire::AverageEffect {
            treatment: q.treatment.raw(),
            outcome: q.outcome.raw(),
            effect_modifiers: vars_to_raw(&q.effect_modifiers),
            control: InterventionWire::from_domain(&q.control)?,
            active: InterventionWire::from_domain(&q.active)?,
            target_population: TargetPopulationWire::from_domain(&q.target_population)?,
        },
        CausalQuery::TemporalEffect(q) => CausalQueryWire::TemporalEffect {
            treatment: q.treatment.raw(),
            outcome: q.outcome.raw(),
            policy: TemporalPolicyWire::from_domain(&q.policy)?,
            control: InterventionWire::from_domain(&q.control)?,
            active: InterventionWire::from_domain(&q.active)?,
            horizon_steps: q.horizon_steps,
            max_history_lag: q.max_history_lag,
            target_population: TargetPopulationWire::from_domain(&q.target_population)?,
        },
        CausalQuery::Counterfactual(q) => CausalQueryWire::Counterfactual {
            outcomes: vars_to_raw(&q.outcomes),
            interventions: q
                .interventions
                .iter()
                .map(InterventionWire::from_domain)
                .collect::<Result<Vec<_>, _>>()?,
            allow_nested: q.allow_nested,
        },
        CausalQuery::AnomalyAttribution(q) => CausalQueryWire::AnomalyAttribution {
            targets: vars_to_raw(&q.targets),
            unit_rows: q
                .unit_rows
                .as_ref()
                .map(|rows| {
                    rows.iter()
                        .map(|&r| u64::try_from(r).map_err(|_| IoError::TooLarge))
                        .collect::<Result<Vec<_>, _>>()
                })
                .transpose()?,
            max_units: u64::try_from(q.max_units).unwrap_or(u64::MAX),
        },
        CausalQuery::ChangeAttribution(q) => CausalQueryWire::ChangeAttribution {
            outcome: q.outcome.raw(),
            baseline: population_to_wire(&q.baseline)?,
            comparison: population_to_wire(&q.comparison)?,
            components: components_to_str(q.components)?.into(),
            allocation: allocation_to_wire(&q.allocation)?,
            max_components: u64::try_from(q.max_components).unwrap_or(u64::MAX),
        },
        CausalQuery::MechanismChange(q) => CausalQueryWire::MechanismChange {
            targets: vars_to_raw(&q.targets),
            baseline: population_to_wire(&q.baseline)?,
            comparison: population_to_wire(&q.comparison)?,
            significance_level_bits: q.significance_level.to_f64().to_bits(),
            max_targets: u64::try_from(q.max_targets).unwrap_or(u64::MAX),
        },
        CausalQuery::UnitChange(q) => CausalQueryWire::UnitChange {
            outcome: q.outcome.raw(),
            unit_rows: q
                .unit_rows
                .as_ref()
                .map(|rows| {
                    rows.iter()
                        .map(|&r| u64::try_from(r).map_err(|_| IoError::TooLarge))
                        .collect::<Result<Vec<_>, _>>()
                })
                .transpose()?,
            components: components_to_str(q.components)?.into(),
            allocation: allocation_to_wire(&q.allocation)?,
            max_units: u64::try_from(q.max_units).unwrap_or(u64::MAX),
        },
        CausalQuery::Mediation(q) => CausalQueryWire::Mediation {
            treatment: q.treatment.raw(),
            outcome: q.outcome.raw(),
            mediators: vars_to_raw(&q.mediators),
            contrast: mediation_contrast_to_str(q.contrast)?.into(),
            control: InterventionWire::from_domain(&q.control)?,
            active: InterventionWire::from_domain(&q.active)?,
            target_population: TargetPopulationWire::from_domain(&q.target_population)?,
        },
        CausalQuery::ConditionalEffect(q) => CausalQueryWire::ConditionalEffect {
            inner: Box::new(causal_query_to_wire(&CausalQuery::AverageEffect(q.inner.clone()))?),
        },
        CausalQuery::Distribution(q) => {
            CausalQueryWire::Distribution(interventional_distribution_to_wire(q)?)
        }
        CausalQuery::PathSpecific(q) => CausalQueryWire::PathSpecific(path_specific_to_wire(q)?),
        _ => return Err(IoError::Convert("unsupported CausalQuery variant".into())),
    })
}

/// Decode [`CausalQueryWire`].
///
/// # Errors
///
/// Unknown tags or size overflows.
pub fn causal_query_from_wire(w: &CausalQueryWire) -> Result<CausalQuery, IoError> {
    Ok(match w {
        CausalQueryWire::AverageEffect {
            treatment,
            outcome,
            effect_modifiers,
            control,
            active,
            target_population,
        } => CausalQuery::AverageEffect(AverageEffectQuery {
            treatment: VariableId::from_raw(*treatment),
            outcome: VariableId::from_raw(*outcome),
            effect_modifiers: vars_from_raw(effect_modifiers),
            control: control.to_domain(),
            active: active.to_domain(),
            target_population: target_population.to_domain()?,
        }),
        CausalQueryWire::TemporalEffect {
            treatment,
            outcome,
            policy,
            control,
            active,
            horizon_steps,
            max_history_lag,
            target_population,
        } => CausalQuery::TemporalEffect(TemporalEffectQuery {
            treatment: VariableId::from_raw(*treatment),
            outcome: VariableId::from_raw(*outcome),
            policy: policy.to_domain(),
            control: control.to_domain(),
            active: active.to_domain(),
            horizon_steps: *horizon_steps,
            max_history_lag: *max_history_lag,
            target_population: target_population.to_domain()?,
        }),
        CausalQueryWire::Counterfactual { outcomes, interventions, allow_nested } => {
            CausalQuery::Counterfactual(CounterfactualQuery {
                outcomes: vars_from_raw(outcomes),
                interventions: interventions.iter().map(InterventionWire::to_domain).collect::<Vec<_>>().into(),
                allow_nested: *allow_nested,
            })
        }
        CausalQueryWire::AnomalyAttribution { targets, unit_rows, max_units } => {
            CausalQuery::AnomalyAttribution(AnomalyAttributionQuery {
                targets: vars_from_raw(targets),
                unit_rows: unit_rows
                    .as_ref()
                    .map(|rows| {
                        rows.iter()
                            .map(|&r| usize::try_from(r).map_err(|_| IoError::TooLarge))
                            .collect::<Result<Vec<_>, _>>()
                    })
                    .transpose()?
                    .map(Arc::from),
                max_units: usize::try_from(*max_units).map_err(|_| IoError::TooLarge)?,
            })
        }
        CausalQueryWire::ChangeAttribution {
            outcome,
            baseline,
            comparison,
            components,
            allocation,
            max_components,
        } => CausalQuery::ChangeAttribution(ChangeAttributionQuery {
            outcome: VariableId::from_raw(*outcome),
            baseline: population_from_wire(baseline)?,
            comparison: population_from_wire(comparison)?,
            components: components_from_str(components)?,
            allocation: allocation_from_wire(allocation)?,
            max_components: usize::try_from(*max_components).map_err(|_| IoError::TooLarge)?,
        }),
        CausalQueryWire::MechanismChange {
            targets,
            baseline,
            comparison,
            significance_level_bits,
            max_targets,
        } => CausalQuery::MechanismChange(MechanismChangeQuery {
            targets: vars_from_raw(targets),
            baseline: population_from_wire(baseline)?,
            comparison: population_from_wire(comparison)?,
            significance_level: OrderedFloatBits::from_f64(f64::from_bits(*significance_level_bits)),
            max_targets: usize::try_from(*max_targets).map_err(|_| IoError::TooLarge)?,
        }),
        CausalQueryWire::UnitChange {
            outcome,
            unit_rows,
            components,
            allocation,
            max_units,
        } => CausalQuery::UnitChange(UnitChangeQuery {
            outcome: VariableId::from_raw(*outcome),
            unit_rows: unit_rows
                .as_ref()
                .map(|rows| {
                    rows.iter()
                        .map(|&r| usize::try_from(r).map_err(|_| IoError::TooLarge))
                        .collect::<Result<Vec<_>, _>>()
                })
                .transpose()?
                .map(Arc::from),
            components: components_from_str(components)?,
            allocation: allocation_from_wire(allocation)?,
            max_units: usize::try_from(*max_units).map_err(|_| IoError::TooLarge)?,
        }),
        CausalQueryWire::Mediation {
            treatment,
            outcome,
            mediators,
            contrast,
            control,
            active,
            target_population,
        } => CausalQuery::Mediation(MediationQuery {
            treatment: VariableId::from_raw(*treatment),
            outcome: VariableId::from_raw(*outcome),
            mediators: vars_from_raw(mediators),
            contrast: mediation_contrast_from_str(contrast)?,
            control: control.to_domain(),
            active: active.to_domain(),
            target_population: target_population.to_domain()?,
        }),
        CausalQueryWire::ConditionalEffect { inner } => {
            let CausalQuery::AverageEffect(inner_q) = causal_query_from_wire(inner)? else {
                return Err(IoError::Convert(
                    "ConditionalEffect.inner must be AverageEffect".into(),
                ));
            };
            CausalQuery::ConditionalEffect(ConditionalEffectQuery { inner: inner_q })
        }
        CausalQueryWire::Distribution(w) => {
            CausalQuery::Distribution(interventional_distribution_from_wire(w)?)
        }
        CausalQueryWire::PathSpecific(w) => CausalQuery::PathSpecific(path_specific_from_wire(w)?),
    })
}

/// Encode an interventional distribution query.
///
/// # Errors
///
/// Unsupported target population.
pub fn interventional_distribution_to_wire(
    q: &InterventionalDistributionQuery,
) -> Result<InterventionalDistributionQueryWire, IoError> {
    Ok(InterventionalDistributionQueryWire {
        outcomes: vars_to_raw(&q.outcomes),
        interventions: q
            .interventions
            .iter()
            .map(InterventionWire::from_domain)
            .collect::<Result<Vec<_>, _>>()?,
        target_population: TargetPopulationWire::from_domain(&q.target_population)?,
    })
}

/// Decode an interventional distribution query.
///
/// # Errors
///
/// Row indices that do not fit `usize`.
pub fn interventional_distribution_from_wire(
    w: &InterventionalDistributionQueryWire,
) -> Result<InterventionalDistributionQuery, IoError> {
    Ok(InterventionalDistributionQuery {
        outcomes: vars_from_raw(&w.outcomes),
        interventions: w.interventions.iter().map(InterventionWire::to_domain).collect::<Vec<_>>().into(),
        target_population: w.target_population.to_domain()?,
    })
}

/// Encode a path-specific effect query.
///
/// # Errors
///
/// Unsupported target population.
pub fn path_specific_to_wire(
    q: &PathSpecificEffectQuery,
) -> Result<PathSpecificEffectQueryWire, IoError> {
    Ok(PathSpecificEffectQueryWire {
        treatment: q.treatment.raw(),
        outcome: q.outcome.raw(),
        path_nodes: vars_to_raw(&q.path_nodes),
        control: InterventionWire::from_domain(&q.control)?,
        active: InterventionWire::from_domain(&q.active)?,
        target_population: TargetPopulationWire::from_domain(&q.target_population)?,
        max_paths: u64::try_from(q.max_paths).unwrap_or(u64::MAX),
        max_len: u64::try_from(q.max_len).unwrap_or(u64::MAX),
    })
}

/// Decode a path-specific effect query.
///
/// # Errors
///
/// Limits that do not fit `usize`.
pub fn path_specific_from_wire(
    w: &PathSpecificEffectQueryWire,
) -> Result<PathSpecificEffectQuery, IoError> {
    Ok(PathSpecificEffectQuery {
        treatment: VariableId::from_raw(w.treatment),
        outcome: VariableId::from_raw(w.outcome),
        path_nodes: vars_from_raw(&w.path_nodes),
        control: w.control.to_domain(),
        active: w.active.to_domain(),
        target_population: w.target_population.to_domain()?,
        max_paths: usize::try_from(w.max_paths)
            .map_err(|_| IoError::Convert("max_paths does not fit usize".into()))?,
        max_len: usize::try_from(w.max_len)
            .map_err(|_| IoError::Convert("max_len does not fit usize".into()))?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::convert::{from_cbor, to_cbor};

    #[test]
    fn average_effect_and_distribution_round_trip() {
        let q = CausalQuery::AverageEffect(AverageEffectQuery::binary_ate(
            VariableId::from_raw(0),
            VariableId::from_raw(1),
        ));
        let wire = causal_query_to_wire(&q).unwrap();
        let bytes = to_cbor(&wire).unwrap();
        let decoded: CausalQueryWire = from_cbor(&bytes).unwrap();
        let back = causal_query_from_wire(&decoded).unwrap();
        assert!(matches!(back, CausalQuery::AverageEffect(_)));
    }

    #[test]
    fn interventional_distribution_cbor_round_trip() {
        let q = InterventionalDistributionQuery::new(
            VariableId::from_raw(1),
            [Intervention::set(VariableId::from_raw(0), Value::f64(3.0))],
        );
        let wire = interventional_distribution_to_wire(&q).unwrap();
        let bytes = to_cbor(&wire).unwrap();
        let decoded: InterventionalDistributionQueryWire = from_cbor(&bytes).unwrap();
        let back = interventional_distribution_from_wire(&decoded).unwrap();
        assert_eq!(back.outcomes.as_ref(), q.outcomes.as_ref());
        assert_eq!(back.interventions.len(), 1);
        back.validate().unwrap();
    }

    #[test]
    fn path_specific_cbor_round_trip() {
        let q = PathSpecificEffectQuery::binary(VariableId::from_raw(0), VariableId::from_raw(2))
            .with_path_nodes([VariableId::from_raw(1)])
            .with_max_paths(32)
            .with_max_len(8);
        let wire = path_specific_to_wire(&q).unwrap();
        let bytes = to_cbor(&wire).unwrap();
        let decoded: PathSpecificEffectQueryWire = from_cbor(&bytes).unwrap();
        let back = path_specific_from_wire(&decoded).unwrap();
        assert_eq!(back.max_paths, 32);
        back.validate().unwrap();
    }

    #[test]
    fn planned_variants_cbor_round_trip() {
        let ate = CausalQuery::AverageEffect(
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1))
                .with_target_population(TargetPopulation::Predicate(PredicateExpr::named(
                    "cohort_a",
                ))),
        );
        let wire = causal_query_to_wire(&ate).unwrap();
        let bytes = to_cbor(&wire).unwrap();
        let decoded: CausalQueryWire = from_cbor(&bytes).unwrap();
        let back = causal_query_from_wire(&decoded).unwrap();
        match back {
            CausalQuery::AverageEffect(q) => match q.target_population {
                TargetPopulation::Predicate(PredicateExpr::Named(name)) => {
                    assert_eq!(&*name, "cohort_a");
                }
                other => panic!("expected PredicateNamed, got {other:?}"),
            },
            other => panic!("expected AverageEffect, got {other:?}"),
        }

        let rows_q = CausalQuery::AverageEffect(
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1))
                .with_target_population(TargetPopulation::Predicate(PredicateExpr::rows([
                    1usize, 3,
                ]))),
        );
        let back = causal_query_from_wire(&causal_query_to_wire(&rows_q).unwrap()).unwrap();
        match back {
            CausalQuery::AverageEffect(q) => match q.target_population {
                TargetPopulation::Predicate(PredicateExpr::Rows(rows)) => {
                    assert_eq!(rows.as_ref(), &[1, 3]);
                }
                other => panic!("expected PredicateRows, got {other:?}"),
            },
            other => panic!("expected AverageEffect, got {other:?}"),
        }

        let dist_q = CausalQuery::AverageEffect(
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1))
                .with_target_population(TargetPopulation::CustomDistribution(
                    DistributionRef::from_raw(9),
                )),
        );
        let back = causal_query_from_wire(&causal_query_to_wire(&dist_q).unwrap()).unwrap();
        match back {
            CausalQuery::AverageEffect(q) => {
                assert_eq!(
                    q.target_population,
                    TargetPopulation::CustomDistribution(DistributionRef::from_raw(9))
                );
            }
            other => panic!("expected AverageEffect, got {other:?}"),
        }

        let temporal = CausalQuery::TemporalEffect(
            TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0)
                .with_policy(TemporalPolicy::dynamic(DynamicRuleId::from_raw(4)))
                .with_horizon_steps(2),
        );
        let back = causal_query_from_wire(&causal_query_to_wire(&temporal).unwrap()).unwrap();
        match back {
            CausalQuery::TemporalEffect(q) => {
                assert_eq!(
                    q.policy,
                    TemporalPolicy::Dynamic {
                        rule: DynamicRuleId::from_raw(4)
                    }
                );
            }
            other => panic!("expected TemporalEffect, got {other:?}"),
        }
    }
}
