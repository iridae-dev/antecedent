//! Feedback between independently declared scalar components.
//!
//! Each interface describes one scalar component's outward value and the
//! inputs its equation reads. Pairwise port checking occurs before the
//! equations are joined; the joined system is then prepared by the existing
//! six-layer cyclic compiler. Its contraction requirement certifies a unique
//! equilibrium for the complete connected system, including feedback that
//! crosses component boundaries. No output inherits an observational or
//! calibrated causal claim from the interface declarations.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use antecedent_contract::evidence_ledger::{EvidenceLedger, Overlap};
use antecedent_contract::interface_contract::{
    CalibrationStatus, ComponentConnection, DependenceRequirement, EvidenceBasis,
    InterfaceContract, InterfaceRefusal, check_connection,
};
use antecedent_core::{CanonicalWriter, ExecutionContext, SemanticDigest};
use antecedent_io::cyclic_system_artifact::{CyclicArtifactError, CyclicExecutionRequest};
use antecedent_model::nonlinear_equilibrium::{
    Activation, Clamp, Equation, EquilibriumModel, SolvePolicy,
};

use super::cyclic::CyclicPreparedState;
use super::{PreparedStudy, Study};
use crate::planner::lowered::{
    EvidenceRole, EvidenceSourceRef, FunctionalKind, IdentificationLayer, LoweredProgram,
    NonGraphicalIdentification, PhysicalLayer, ProblemTarget, ProgramCell,
    StandaloneCompositionLoweredParts,
};
use crate::planner::{PlanStage, PlanView};

const COMPOSITION_TAG: &str = "antecedent.cyclic_component_composition.v1";

fn activation_tag(activation: Activation) -> u64 {
    match activation {
        Activation::Identity => 1,
        Activation::Tanh => 2,
        Activation::Sin => 3,
        Activation::Relu => 4,
    }
}

/// One scalar structural mechanism with a checked scientific interface.
/// The local equation reads its own value through its sole weight; incoming
/// component weights are supplied by [`FeedbackConnection`].
#[derive(Clone, Debug)]
pub struct ScalarFeedbackComponent {
    /// Exactly one produced port names this component's endogenous value.
    pub interface: InterfaceContract,
    /// Local scalar equation, before cross-component connections are added.
    pub equation: Equation,
}

/// A checked producer-to-consumer port and its structural coefficient.
#[derive(Clone, Debug)]
pub struct FeedbackConnection {
    /// The scientific port declaration to be checked.
    pub ports: ComponentConnection,
    /// Coefficient in the consumer's structural equation for the producer.
    pub coefficient: f64,
}

/// Hard-clamp interventions named by component, rather than by dense index.
#[derive(Clone, Debug)]
pub struct ComponentFeedbackWorld {
    /// Each component is clamped at most once in this intervention world.
    pub clamps: Vec<(String, f64)>,
}

/// A complete model-conditional request for a connected equilibrium.
#[derive(Clone, Debug)]
pub struct ComponentFeedbackRequest {
    /// Scalar mechanisms with their scientific ports.
    pub components: Vec<ScalarFeedbackComponent>,
    /// Structural coefficients between components.
    pub connections: Vec<FeedbackConnection>,
    /// Full row-major covariance of joint disturbances in sorted component-name order.
    pub noise_covariance: Vec<f64>,
    /// Explicit aligned disturbance realizations in that order, shared across worlds.
    pub noise: Vec<Vec<f64>>,
    /// Numerical starting state in sorted component-name order.
    pub initial: Vec<f64>,
    /// Interventions, each a set of hard clamps.
    pub worlds: Vec<ComponentFeedbackWorld>,
    /// A complete-system solve policy. Global contraction is required.
    pub policy: SolvePolicy,
    /// Optional factual state for a shared-noise counterfactual.
    pub factual: Option<Vec<f64>>,
}

/// One output equation of a component with several produced ports. Each
/// consumed port may be read by several outputs, with a separate coefficient
/// for each output equation.
#[derive(Clone, Debug)]
pub struct VectorFeedbackOutput {
    /// Name of the produced scientific port.
    pub produced_port: String,
    /// Local scalar equation, with one self coefficient before connections.
    pub equation: Equation,
    /// Consumed port names read by this output equation.
    pub consumed_ports: Vec<String>,
}

/// A group of output equations sharing one scientific component interface.
#[derive(Clone, Debug)]
pub struct VectorFeedbackComponent {
    /// Every produced and consumed port of this component.
    pub interface: InterfaceContract,
    /// Exactly one output equation per produced port.
    pub outputs: Vec<VectorFeedbackOutput>,
}

/// A port connection aimed at one named output equation of a multi-output
/// consumer. The same consumed port may feed several outputs with different
/// coefficients.
#[derive(Clone, Debug)]
pub struct VectorFeedbackConnection {
    /// Producer, produced port, consumer and consumed port declarations.
    pub ports: ComponentConnection,
    /// Produced port of the consumer whose equation reads this connection.
    pub consumer_output: String,
    /// Coefficient in that output equation.
    pub coefficient: f64,
}

/// A connected multi-output system. Disturbance covariance, aligned noise,
/// initial and factual states use sorted `component::produced_port` order.
/// World clamps name those complete output identities.
#[derive(Clone, Debug)]
pub struct VectorFeedbackRequest {
    /// Components with one or more produced outputs.
    pub components: Vec<VectorFeedbackComponent>,
    /// Connections between their declared ports.
    pub connections: Vec<VectorFeedbackConnection>,
    /// Full row-major covariance in sorted output order.
    pub noise_covariance: Vec<f64>,
    /// Joint disturbance draws in sorted output order.
    pub noise: Vec<Vec<f64>>,
    /// Initial equilibrium state in sorted output order.
    pub initial: Vec<f64>,
    /// Named hard-clamp worlds.
    pub worlds: Vec<ComponentFeedbackWorld>,
    /// Complete-system solve policy; global contraction remains required.
    pub policy: SolvePolicy,
    /// Optional factual state in sorted output order.
    pub factual: Option<Vec<f64>>,
}

/// Compile several output equations per component into one connected cyclic
/// program. The expansion preserves scientific port checks and gives each
/// output a distinct structural coordinate. It does not promote component
/// evidence or calibration to the whole equilibrium.
///
/// # Errors
/// A missing or repeated output, invalid connection or any
/// refusal of [`prepare_component_feedback`].
pub fn prepare_vector_component_feedback(
    request: VectorFeedbackRequest,
    ctx: &ExecutionContext,
) -> Result<PreparedComponentFeedback, ComponentFeedbackError> {
    prepare_vector_component_feedback_inner(request, None, ctx)
}

/// Compile multi-output feedback with a ledger check over the original
/// component's inputs, including inputs read by different output equations.
/// The ledger state is bound into the central cyclic program identity.
pub fn prepare_vector_component_feedback_with_ledger(
    request: VectorFeedbackRequest,
    ledger: &EvidenceLedger,
    ctx: &ExecutionContext,
) -> Result<PreparedComponentFeedback, ComponentFeedbackError> {
    prepare_vector_component_feedback_inner(request, Some(ledger), ctx)
}

fn prepare_vector_component_feedback_inner(
    request: VectorFeedbackRequest,
    ledger: Option<&EvidenceLedger>,
    ctx: &ExecutionContext,
) -> Result<PreparedComponentFeedback, ComponentFeedbackError> {
    check_aggregate_evidence(
        &request.components.iter().map(|component| &component.interface).collect::<Vec<_>>(),
        &request.connections.iter().map(|connection| &connection.ports).collect::<Vec<_>>(),
        ledger,
    )?;
    let mut produced = BTreeMap::<(String, String), String>::new();
    let mut consumed = BTreeMap::<(String, String, String), String>::new();
    let mut scalar = Vec::new();
    for component in request.components {
        component
            .interface
            .validate()
            .map_err(|refusal| ComponentFeedbackError::Interface(vec![refusal]))?;
        if component.outputs.len() != component.interface.produces.len() {
            return Err(ComponentFeedbackError::Invalid(
                "each produced port needs exactly one output equation",
            ));
        }
        let name = component.interface.component.clone();
        let mut seen_outputs = BTreeSet::new();
        for output in component.outputs {
            if !seen_outputs.insert(output.produced_port.clone()) {
                return Err(ComponentFeedbackError::Invalid("an output equation is repeated"));
            }
            let port = component
                .interface
                .produces
                .iter()
                .find(|port| port.name == output.produced_port)
                .ok_or(ComponentFeedbackError::Invalid(
                    "an output equation names no produced port",
                ))?;
            let full = format!("{name}::{}", port.name);
            if produced.insert((name.clone(), port.name.clone()), full.clone()).is_some() {
                return Err(ComponentFeedbackError::Invalid("a produced port is repeated"));
            }
            let mut inputs = Vec::new();
            for input_name in &output.consumed_ports {
                let input = component
                    .interface
                    .consumes
                    .iter()
                    .find(|input| input.name == *input_name)
                    .ok_or(ComponentFeedbackError::Invalid(
                        "an output reads an undeclared consumed port",
                    ))?;
                if consumed
                    .insert((name.clone(), port.name.clone(), input.name.clone()), full.clone())
                    .is_some()
                {
                    return Err(ComponentFeedbackError::Invalid(
                        "a consumed port is repeated within one output equation",
                    ));
                }
                inputs.push(input.clone());
            }
            scalar.push(ScalarFeedbackComponent {
                interface: InterfaceContract {
                    component: full,
                    produces: vec![port.clone()],
                    consumes: inputs,
                },
                equation: output.equation,
            });
        }
        if component.interface.consumes.iter().any(|port| {
            !consumed.keys().any(|(component, _, input)| component == &name && input == &port.name)
        }) {
            return Err(ComponentFeedbackError::Invalid(
                "every consumed port needs one output equation",
            ));
        }
    }
    let mut connections = Vec::with_capacity(request.connections.len());
    for connection in request.connections {
        let source = produced
            .get(&(connection.ports.producer.clone(), connection.ports.produced_port.clone()))
            .ok_or(ComponentFeedbackError::Invalid("a connection names no produced output"))?;
        let target = consumed
            .get(&(
                connection.ports.consumer.clone(),
                connection.consumer_output.clone(),
                connection.ports.consumed_port.clone(),
            ))
            .ok_or(ComponentFeedbackError::Invalid("a connection names no consumed output"))?;
        connections.push(FeedbackConnection {
            ports: ComponentConnection::new(
                source.clone(),
                connection.ports.produced_port.clone(),
                target.clone(),
                connection.ports.consumed_port.clone(),
            ),
            coefficient: connection.coefficient,
        });
    }
    prepare_component_feedback_inner(
        ComponentFeedbackRequest {
            components: scalar,
            connections,
            noise_covariance: request.noise_covariance,
            noise: request.noise,
            initial: request.initial,
            worlds: request.worlds,
            policy: request.policy,
            factual: request.factual,
        },
        ledger,
        ctx,
    )
}

fn check_aggregate_evidence(
    components: &[&InterfaceContract],
    connections: &[&ComponentConnection],
    ledger: Option<&EvidenceLedger>,
) -> Result<(), ComponentFeedbackError> {
    let by_name: BTreeMap<&str, &InterfaceContract> =
        components.iter().map(|contract| (contract.component.as_str(), *contract)).collect();
    for consumer in components {
        let inputs: Vec<_> = consumer
            .consumes
            .iter()
            .filter(|port| port.dependence == DependenceRequirement::DisjointEvidence)
            .collect();
        for (index, first) in inputs.iter().enumerate() {
            for second in &inputs[index + 1..] {
                let sources = |port_name: &str| -> Result<_, ComponentFeedbackError> {
                    let mut incoming = connections.iter().filter(|connection| {
                        connection.consumer == consumer.component
                            && connection.consumed_port == port_name
                    });
                    let connection = incoming.next().ok_or(ComponentFeedbackError::Invalid(
                        "a disjoint-evidence input has no producer",
                    ))?;
                    if incoming.next().is_some() {
                        return Err(ComponentFeedbackError::Invalid(
                            "a disjoint-evidence input is multiply fed",
                        ));
                    }
                    let producer = by_name.get(connection.producer.as_str()).ok_or(
                        ComponentFeedbackError::Invalid("an evidence producer is unknown"),
                    )?;
                    let output = producer
                        .produces
                        .iter()
                        .find(|port| port.name == connection.produced_port)
                        .ok_or(ComponentFeedbackError::Invalid("an evidence output is unknown"))?;
                    match &output.evidence {
                        EvidenceBasis::Declared(sources) if !sources.is_empty() => {
                            Ok(sources.clone())
                        }
                        _ => Err(ComponentFeedbackError::Invalid(
                            "disjoint-evidence inputs need declared producer sources",
                        )),
                    }
                };
                let (left, right) = (sources(&first.name)?, sources(&second.name)?);
                let ledger = ledger.ok_or(ComponentFeedbackError::Invalid(
                    "multiple disjoint-evidence inputs require an evidence ledger",
                ))?;
                for &a in &left {
                    for &b in &right {
                        if a == b
                            || !ledger.source(a).is_some_and(|source| source.status.is_active())
                            || !ledger.source(b).is_some_and(|source| source.status.is_active())
                            || !matches!(ledger.overlap(a, b), Ok(Overlap::Independent))
                        {
                            return Err(ComponentFeedbackError::Invalid(
                                "the aggregate ledger does not prove input evidence disjoint",
                            ));
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

/// The interface check or central cyclic compiler refused the system.
#[derive(Debug, thiserror::Error)]
pub enum ComponentFeedbackError {
    /// A malformed component declaration or coupling.
    #[error("cyclic component composition: {0}")]
    Invalid(&'static str),
    /// Scientific port compatibility failed.
    #[error("cyclic component interface: {0:?}")]
    Interface(Vec<InterfaceRefusal>),
    /// The central cyclic compiler or solver refused the connected system.
    #[error("cyclic component compiler: {0}")]
    Cyclic(#[from] CyclicArtifactError),
}

/// A connected system prepared through the central cyclic compiler.
pub struct PreparedComponentFeedback {
    prepared: PreparedStudy<CyclicPreparedState>,
    order: Vec<String>,
    composition_identity: SemanticDigest,
}

impl PreparedComponentFeedback {
    /// Retained six-layer cyclic handle. Execution and export use this handle's
    /// normal compiler-routed methods and independent artifact consumer.
    #[must_use]
    pub const fn prepared(&self) -> &PreparedStudy<CyclicPreparedState> {
        &self.prepared
    }

    /// Dense variable order of the complete system.
    #[must_use]
    pub fn component_order(&self) -> &[String] {
        &self.order
    }

    /// Identity of the scientific interfaces and structural couplings.
    #[must_use]
    pub const fn composition_identity(&self) -> SemanticDigest {
        self.composition_identity
    }
}

/// Compile scalar components as one connected cyclic system. Interface
/// compatibility and the complete-system contraction requirement are checked
/// before any result is reported. Cross-component cycles are solved as cycles;
/// no feed-forward order or finite unrolling is used.
///
/// # Errors
/// A missing or multiply fed port, malformed equation or intervention, failed
/// scientific interface check, or refusal of the central cyclic compiler.
pub fn prepare_component_feedback(
    request: ComponentFeedbackRequest,
    ctx: &ExecutionContext,
) -> Result<PreparedComponentFeedback, ComponentFeedbackError> {
    prepare_component_feedback_inner(request, None, ctx)
}

/// Compile scalar feedback with a checked aggregate evidence ledger. The
/// ledger's state digest enters the six-layer cyclic program through the
/// structural equation identities.
pub fn prepare_component_feedback_with_ledger(
    request: ComponentFeedbackRequest,
    ledger: &EvidenceLedger,
    ctx: &ExecutionContext,
) -> Result<PreparedComponentFeedback, ComponentFeedbackError> {
    prepare_component_feedback_inner(request, Some(ledger), ctx)
}

fn prepare_component_feedback_inner(
    mut request: ComponentFeedbackRequest,
    ledger: Option<&EvidenceLedger>,
    ctx: &ExecutionContext,
) -> Result<PreparedComponentFeedback, ComponentFeedbackError> {
    check_aggregate_evidence(
        &request.components.iter().map(|component| &component.interface).collect::<Vec<_>>(),
        &request.connections.iter().map(|connection| &connection.ports).collect::<Vec<_>>(),
        ledger,
    )?;
    if request.components.is_empty() || !request.policy.require_contraction {
        return Err(ComponentFeedbackError::Invalid(
            "at least one component and a global contraction certificate are required",
        ));
    }
    request
        .components
        .sort_by(|left, right| left.interface.component.cmp(&right.interface.component));
    let order: Vec<String> =
        request.components.iter().map(|component| component.interface.component.clone()).collect();
    if order.iter().any(|name| name.trim().is_empty())
        || order.windows(2).any(|pair| pair[0] == pair[1])
        || request.initial.len() != order.len()
        || request.noise_covariance.len() != order.len().saturating_mul(order.len())
        || request.factual.as_ref().is_some_and(|factual| factual.len() != order.len())
        || request.noise.iter().any(|row| row.len() != order.len())
    {
        return Err(ComponentFeedbackError::Invalid(
            "component identities, disturbance covariance, initial state or factual state have an invalid shape",
        ));
    }
    let positions: BTreeMap<&str, usize> =
        order.iter().enumerate().map(|(index, name)| (name.as_str(), index)).collect();
    let mut writer = CanonicalWriter::new(COMPOSITION_TAG);
    writer.word(order.len() as u64);
    if let Some(ledger) = ledger {
        writer.bytes(ledger.state_digest().as_bytes());
    }
    for component in &request.components {
        component
            .interface
            .validate()
            .map_err(|refusal| ComponentFeedbackError::Interface(vec![refusal]))?;
        if component.interface.produces.len() != 1
            || component.equation.weights.len() != 1
            || component.equation.expression.is_some()
            || !component.equation.parameters.is_empty()
        {
            return Err(ComponentFeedbackError::Invalid(
                "each component needs one produced port and one scalar local equation",
            ));
        }
        component
            .equation
            .validate(1)
            .map_err(|_| ComponentFeedbackError::Invalid("a local scalar equation is invalid"))?;
        if component.interface.produces.iter().any(|port| {
            matches!(
                &port.calibration,
                CalibrationStatus::Measured { .. } | CalibrationStatus::Inherited { .. }
            )
        }) {
            return Err(ComponentFeedbackError::Invalid(
                "a measured or inherited component calibration does not cover the connected equilibrium",
            ));
        }
        if component
            .interface
            .consumes
            .iter()
            .filter(|port| port.dependence == DependenceRequirement::DisjointEvidence)
            .count()
            > 1
        {
            return Err(ComponentFeedbackError::Invalid(
                "multiple disjoint-evidence inputs require a joint evidence-ledger check",
            ));
        }
        writer
            .text(&component.interface.component)
            .text(&component.interface.canonical_identity().to_hex())
            .text(&component.equation.id)
            .word(component.equation.intercept.to_bits())
            .word(component.equation.weights[0].to_bits())
            .word(activation_tag(component.equation.activation));
    }
    request.connections.sort_by(|left, right| {
        (
            &left.ports.consumer,
            &left.ports.consumed_port,
            &left.ports.producer,
            &left.ports.produced_port,
        )
            .cmp(&(
                &right.ports.consumer,
                &right.ports.consumed_port,
                &right.ports.producer,
                &right.ports.produced_port,
            ))
    });
    let mut fed = BTreeSet::new();
    let mut matrix_edges = BTreeSet::new();
    let mut couplings = Vec::with_capacity(request.connections.len());
    for connection in &request.connections {
        let ports = &connection.ports;
        let Some(&producer) = positions.get(ports.producer.as_str()) else {
            return Err(ComponentFeedbackError::Invalid("a connection names an unknown producer"));
        };
        let Some(&consumer) = positions.get(ports.consumer.as_str()) else {
            return Err(ComponentFeedbackError::Invalid("a connection names an unknown consumer"));
        };
        if producer == consumer
            || !connection.coefficient.is_finite()
            || connection.coefficient == 0.0
            || !fed.insert((consumer, ports.consumed_port.as_str()))
            || !matrix_edges.insert((consumer, producer))
        {
            return Err(ComponentFeedbackError::Invalid(
                "a cross-component input must be distinct, nonzero and occupy one structural coefficient",
            ));
        }
        let checked = check_connection(
            &request.components[producer].interface,
            &ports.produced_port,
            &request.components[consumer].interface,
            &ports.consumed_port,
        )
        .map_err(ComponentFeedbackError::Interface)?;
        writer.text(&checked.identity.to_hex()).word(connection.coefficient.to_bits());
        couplings.push((consumer, producer, connection.coefficient));
    }
    for (index, component) in request.components.iter().enumerate() {
        if component
            .interface
            .consumes
            .iter()
            .any(|port| !fed.contains(&(index, port.name.as_str())))
        {
            return Err(ComponentFeedbackError::Invalid(
                "every declared consumed port needs one structural connection",
            ));
        }
    }
    let composition_identity = SemanticDigest::from_bytes(writer.finish());
    let n = order.len();
    let mut equations: Vec<Equation> = request
        .components
        .into_iter()
        .enumerate()
        .map(|(index, component)| {
            let mut equation = component.equation;
            let self_weight = equation.weights[0];
            equation.id = format!("component.{index}.{}", composition_identity.to_hex());
            equation.weights = vec![0.0; n];
            equation.weights[index] = self_weight;
            equation
        })
        .collect();
    for (consumer, producer, coefficient) in couplings {
        equations[consumer].weights[producer] = coefficient;
    }
    let worlds = request
        .worlds
        .into_iter()
        .map(|world| {
            let mut seen = BTreeSet::new();
            world
                .clamps
                .into_iter()
                .map(|(name, value)| {
                    let variable = *positions.get(name.as_str()).ok_or(
                        ComponentFeedbackError::Invalid("a clamp names an unknown component"),
                    )?;
                    if !value.is_finite() || !seen.insert(variable) {
                        return Err(ComponentFeedbackError::Invalid(
                            "a world clamps each component at most once to a finite value",
                        ));
                    }
                    Ok(Clamp { variable, value })
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?;
    let prepared = Study::prepare_cyclic(
        CyclicExecutionRequest {
            model: EquilibriumModel { equations, noise_covariance: request.noise_covariance },
            fit: None,
            worlds,
            noise: request.noise,
            initial: request.initial,
            policy: request.policy,
            factual: request.factual,
        },
        ctx,
    )?;
    Ok(PreparedComponentFeedback { prepared, order, composition_identity })
}

/// Shared raw evidence and sampler schedule for one connected-model fit.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ComponentJointPosteriorSchedule {
    /// Jointly fitted coordinates in sorted component order.
    pub parameters: Vec<antecedent_estimate::cyclic_likelihood::CyclicParameter>,
    /// Common intervention-labelled rows for the complete system.
    pub training: Vec<antecedent_model::nonlinear_equilibrium::RegimeRow>,
    /// Shared independent normal parameter prior scale.
    pub prior_sd: f64,
    /// Independent MCMC chains.
    pub chains: usize,
    /// Discarded draws per chain.
    pub warmup: usize,
    /// Retained draws per chain.
    pub draws_per_chain: usize,
    /// Proposal standard deviation.
    pub proposal_scale: f64,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct ComponentConnectionWire {
    producer: String,
    produced_port: String,
    consumer: String,
    consumed_port: String,
    coefficient: f64,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct ComponentWorldWire {
    clamps: Vec<(String, f64)>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct ComponentFeedbackWire {
    components: Vec<(antecedent_io::interface_contract_artifact::ContractWire, Equation)>,
    connections: Vec<ComponentConnectionWire>,
    noise_covariance: Vec<f64>,
    noise: Vec<Vec<f64>>,
    initial: Vec<f64>,
    worlds: Vec<ComponentWorldWire>,
    policy: SolvePolicy,
}

impl ComponentFeedbackWire {
    fn from_request(request: &ComponentFeedbackRequest) -> Self {
        Self {
            components: request
                .components
                .iter()
                .map(|component| {
                    (
                        antecedent_io::interface_contract_artifact::contract_to_wire(
                            &component.interface,
                        ),
                        component.equation.clone(),
                    )
                })
                .collect(),
            connections: request
                .connections
                .iter()
                .map(|connection| ComponentConnectionWire {
                    producer: connection.ports.producer.clone(),
                    produced_port: connection.ports.produced_port.clone(),
                    consumer: connection.ports.consumer.clone(),
                    consumed_port: connection.ports.consumed_port.clone(),
                    coefficient: connection.coefficient,
                })
                .collect(),
            noise_covariance: request.noise_covariance.clone(),
            noise: request.noise.clone(),
            initial: request.initial.clone(),
            worlds: request
                .worlds
                .iter()
                .map(|world| ComponentWorldWire { clamps: world.clamps.clone() })
                .collect(),
            policy: request.policy,
        }
    }

    fn to_request(&self) -> Result<ComponentFeedbackRequest, JointComponentPosteriorError> {
        let components = self
            .components
            .iter()
            .map(|(wire, equation)| {
                let interface = antecedent_io::interface_contract_artifact::contract_from_wire(
                    wire,
                )
                .map_err(|_| {
                    JointComponentPosteriorError::Invalid("invalid component interface wire")
                })?;
                Ok(ScalarFeedbackComponent { interface, equation: equation.clone() })
            })
            .collect::<Result<Vec<_>, JointComponentPosteriorError>>()?;
        Ok(ComponentFeedbackRequest {
            components,
            connections: self
                .connections
                .iter()
                .map(|connection| FeedbackConnection {
                    ports: ComponentConnection::new(
                        connection.producer.as_str(),
                        connection.produced_port.as_str(),
                        connection.consumer.as_str(),
                        connection.consumed_port.as_str(),
                    ),
                    coefficient: connection.coefficient,
                })
                .collect(),
            noise_covariance: self.noise_covariance.clone(),
            noise: self.noise.clone(),
            initial: self.initial.clone(),
            worlds: self
                .worlds
                .iter()
                .map(|world| ComponentFeedbackWorld { clamps: world.clamps.clone() })
                .collect(),
            policy: self.policy,
            factual: None,
        })
    }
}

/// Refusal of a checked connected-component posterior handoff.
#[derive(Debug, thiserror::Error)]
pub enum JointComponentPosteriorError {
    /// Component interfaces, connections or cyclic preparation refused.
    #[error("connected component posterior: {0}")]
    Component(#[from] ComponentFeedbackError),
    /// The jointly fitted posterior or its decision refused.
    #[error("connected component posterior: {0}")]
    Posterior(#[from] super::cyclic_posterior_action_decision::CyclicPosteriorDecisionError),
    /// Portable encoding or decoding failed.
    #[error("connected component posterior artifact: {0}")]
    Io(#[from] antecedent_io::IoError),
    /// An unsupported or inconsistent binding was supplied.
    #[error("connected component posterior: {0}")]
    Invalid(&'static str),
}

/// Checked component graph and its shared-data, jointly fitted action program.
#[derive(Clone, Debug)]
pub struct CompiledJointComponentPosteriorDecision {
    component_wire: ComponentFeedbackWire,
    schedule: ComponentJointPosteriorSchedule,
    outputs: Vec<super::cyclic_action_world_decision::CyclicDecisionOutput>,
    contract: antecedent_design::decision_contract::DecisionContract,
    posterior: super::cyclic_posterior_action_decision::CompiledCyclicPosteriorDecision,
    lowered: Arc<LoweredProgram>,
}

/// A fitted child decision and the complete component compiler identity.
#[derive(Clone, Debug)]
pub struct JointComponentPosteriorDecisionResult {
    /// The fitted posterior and decision produced by the child program.
    pub child: super::cyclic_posterior_action_decision::CyclicPosteriorDecisionResult,
    /// Identity of the outer program binding the component wire and child.
    pub lowered_program_identity: SemanticDigest,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct JointComponentPosteriorWire {
    version: u32,
    required_feature: String,
    components: ComponentFeedbackWire,
    schedule: ComponentJointPosteriorSchedule,
    outputs: Vec<super::cyclic_action_world_decision::CyclicDecisionOutput>,
    contract_artifact: Vec<u8>,
    contract_identity: String,
    program_identity: [u8; 32],
    posterior_artifact: Vec<u8>,
}

const JOINT_COMPONENT_POSTERIOR_FEATURE: &str = "joint_component_posterior_decision_v1";
const MAX_JOINT_COMPONENT_BYTES: usize = 80 * 1024 * 1024;

/// Compile checked connected components, one common raw evidence schedule,
/// the joint covariance and action worlds through the cyclic posterior route.
/// The component-preparation noise must be one zero placeholder row; all
/// predictive disturbances are instead generated jointly from its covariance.
///
/// # Errors
/// Invalid component ports, coordinate bindings, evidence or posterior schedule.
pub fn compile_joint_component_posterior_decision(
    request: ComponentFeedbackRequest,
    schedule: ComponentJointPosteriorSchedule,
    outputs: Vec<super::cyclic_action_world_decision::CyclicDecisionOutput>,
    contract: antecedent_design::decision_contract::DecisionContract,
    ctx: &ExecutionContext,
) -> Result<CompiledJointComponentPosteriorDecision, JointComponentPosteriorError> {
    if request.factual.is_some()
        || request.noise.len() != 1
        || request.noise[0].iter().any(|value| *value != 0.0)
    {
        return Err(JointComponentPosteriorError::Invalid(
            "posterior handoff requires one zero placeholder noise row and no factual state",
        ));
    }
    if request.components.iter().any(|component| {
        component.interface.produces.iter().any(|port| {
            !matches!(port.support, antecedent_contract::interface_contract::SupportDomain::Real)
        }) || component.interface.consumes.iter().any(|port| {
            !matches!(port.support, antecedent_contract::interface_contract::SupportDomain::Real)
        })
    }) {
        return Err(JointComponentPosteriorError::Invalid(
            "joint posterior handoff requires finite real component ports",
        ));
    }
    let wire = ComponentFeedbackWire::from_request(&request);
    let declarations = request.components.clone();
    let prepared = prepare_component_feedback(request, ctx)?;
    for output in &outputs {
        let name = prepared
            .order
            .get(output.equation)
            .ok_or(JointComponentPosteriorError::Invalid("output component is absent"))?;
        let component = declarations
            .iter()
            .find(|component| &component.interface.component == name)
            .ok_or(JointComponentPosteriorError::Invalid("output component interface is absent"))?;
        let port = component
            .interface
            .produces
            .first()
            .ok_or(JointComponentPosteriorError::Invalid("output port is absent"))?;
        let quantity: antecedent_contract::ScientificQuantity =
            output.quantity.clone().try_into().map_err(|_| {
                JointComponentPosteriorError::Invalid("output coordinate is invalid")
            })?;
        let mut declared = port.quantity.clone();
        // Each action world changes regime; all other scientific coordinate
        // fields, including variable identity, come from the checked port.
        declared.regime_id = quantity.regime_id.clone();
        declared.require_same_coordinate(&quantity).map_err(|_| {
            JointComponentPosteriorError::Invalid("output coordinate differs from component port")
        })?;
    }
    let cyclic = prepared.prepared.request();
    let posterior_request =
        super::cyclic_posterior_action_decision::CyclicPosteriorDecisionRequest {
            model: cyclic.model.clone(),
            parameters: schedule.parameters.clone(),
            training: schedule.training.clone(),
            prior_sd: schedule.prior_sd,
            chains: schedule.chains,
            warmup: schedule.warmup,
            draws_per_chain: schedule.draws_per_chain,
            proposal_scale: schedule.proposal_scale,
            worlds: cyclic.worlds.clone(),
            initial: cyclic.initial.clone(),
            solve_policy: cyclic.policy,
        };
    let posterior =
        super::cyclic_posterior_action_decision::compile_cyclic_component_posterior_decision(
            posterior_request,
            outputs.clone(),
            contract.clone(),
            ctx,
        )?;
    let component_bytes = antecedent_io::to_cbor(&wire)?;
    let component_digest = CanonicalWriter::digest_bytes(
        "antecedent.joint_component_posterior.component_wire.v1",
        &component_bytes,
    );
    let mut writer = CanonicalWriter::new("antecedent.joint_component_posterior.question.v1");
    writer
        .text(JOINT_COMPONENT_POSTERIOR_FEATURE)
        .bytes(&component_digest)
        .bytes(posterior.lowered().identity().as_bytes())
        .bytes(&antecedent_io::to_cbor(&outputs)?);
    let query = writer.finish();
    let mut scientific = posterior.lowered().scientific().clone();
    scientific.target = ProblemTarget::QueryFunctional {
        kind: FunctionalKind::PropagatedChain,
        outcome: None,
        query,
    };
    scientific.plan_id = Arc::from("joint_component_posterior_decision.v1");
    scientific.source_digest = Some(component_digest);
    let identification = IdentificationLayer::NonGraphical(Arc::new(
        NonGraphicalIdentification::SharedEvidenceContract { contract: query },
    ));
    let mut evidence = posterior.lowered().evidence().clone();
    let mut sources = evidence.sources.to_vec();
    sources.push(EvidenceSourceRef {
        role: EvidenceRole::ChainSource,
        digest: Some(component_digest),
    });
    evidence.sources = Arc::from(sources);
    let mut operators = posterior.lowered().physical().operators().to_vec();
    operators.push(
        antecedent_core::PhysicalOperator::declare(antecedent_core::PhysicalEffect::Read)
            .estimates(
                antecedent_core::ByteEstimates::absent().with_output(component_bytes.len() as u64),
            )
            .build()
            .map_err(|_| {
                JointComponentPosteriorError::Invalid("component wire read cannot be lowered")
            })?,
    );
    let physical = PhysicalLayer {
        operators: Arc::from(operators),
        design: posterior.lowered().physical().design(),
    };
    let lowered = Arc::new(
        StandaloneCompositionLoweredParts {
            cell: ProgramCell::JointComponentPosteriorDecision,
            scientific,
            identification,
            statistical: posterior.lowered().statistical().clone(),
            evidence,
            physical,
            claim: *posterior.lowered().claim(),
        }
        .finish(),
    );
    Ok(CompiledJointComponentPosteriorDecision {
        component_wire: wire,
        schedule,
        outputs,
        contract,
        posterior,
        lowered,
    })
}

impl CompiledJointComponentPosteriorDecision {
    /// Outer six-layer program binding the full checked component declaration.
    #[must_use]
    pub fn lowered(&self) -> &LoweredProgram {
        &self.lowered
    }

    /// Inspect the whole compiled route without fitting.
    #[must_use]
    pub fn plan_view(&self) -> PlanView {
        PlanView::of_program(Arc::clone(&self.lowered), PlanStage::Compiled)
    }

    /// Refit the common evidence schedule and solve all paired action worlds.
    /// Connected calibration is unmeasured.
    ///
    /// # Errors
    /// Fit, posterior diagnostics, equilibrium or decision refusal.
    pub fn execute(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<JointComponentPosteriorDecisionResult, JointComponentPosteriorError> {
        let scoped = ctx.enter_lowered_dispatch(self.lowered.identity());
        let child = self.posterior.execute(&scoped)?;
        Ok(JointComponentPosteriorDecisionResult {
            child,
            lowered_program_identity: self.lowered.identity(),
        })
    }

    /// Reexecute and seal both the checked component declarations and joint fit.
    ///
    /// # Errors
    /// A changed result or oversized portable artifact.
    pub fn export(
        &self,
        result: &JointComponentPosteriorDecisionResult,
        ctx: &ExecutionContext,
    ) -> Result<Vec<u8>, JointComponentPosteriorError> {
        if result.lowered_program_identity != self.lowered.identity() {
            return Err(JointComponentPosteriorError::Invalid(
                "component execution program differs",
            ));
        }
        let scoped = ctx.enter_lowered_dispatch(self.lowered.identity());
        let contract = antecedent_design::decision_artifact::DecisionContractArtifact::new(
            self.contract.clone(),
        )
        .map_err(super::cyclic_posterior_action_decision::CyclicPosteriorDecisionError::from)?;
        let bytes = antecedent_io::to_cbor(&JointComponentPosteriorWire {
            version: 1,
            required_feature: JOINT_COMPONENT_POSTERIOR_FEATURE.into(),
            components: self.component_wire.clone(),
            schedule: self.schedule.clone(),
            outputs: self.outputs.clone(),
            contract_identity: contract.identity().into(),
            contract_artifact: contract.to_bytes("joint-component-contract")?,
            program_identity: *self.lowered().identity().as_bytes(),
            posterior_artifact: self.posterior.export(
                &result.child,
                "joint-component-posterior",
                &scoped,
            )?,
        })?;
        if bytes.len() > MAX_JOINT_COMPONENT_BYTES {
            return Err(JointComponentPosteriorError::Invalid(
                "portable component posterior exceeds its bound",
            ));
        }
        Ok(bytes)
    }
}

/// Recheck the component interfaces and coupling, recompile the joint posterior,
/// refit shared raw evidence and independently verify the portable result.
///
/// # Errors
/// Changed feature, component binding, compiler identity, fit or decision.
pub fn consume_joint_component_posterior_decision(
    bytes: &[u8],
    expected_program: SemanticDigest,
    ctx: &ExecutionContext,
) -> Result<JointComponentPosteriorDecisionResult, JointComponentPosteriorError> {
    if bytes.len() > MAX_JOINT_COMPONENT_BYTES {
        return Err(JointComponentPosteriorError::Invalid(
            "portable component posterior exceeds its bound",
        ));
    }
    let wire: JointComponentPosteriorWire = antecedent_io::from_cbor(bytes)?;
    if wire.version != 1
        || wire.required_feature != JOINT_COMPONENT_POSTERIOR_FEATURE
        || wire.program_identity != *expected_program.as_bytes()
    {
        return Err(JointComponentPosteriorError::Invalid(
            "component posterior feature or program differs",
        ));
    }
    let contract = antecedent_design::decision_artifact::DecisionContractArtifact::from_bytes(
        &wire.contract_artifact,
        &wire.contract_identity,
    )?;
    let plan = compile_joint_component_posterior_decision(
        wire.components.to_request()?,
        wire.schedule,
        wire.outputs,
        contract.contract().clone(),
        ctx,
    )?;
    if plan.lowered().identity() != expected_program {
        return Err(JointComponentPosteriorError::Invalid(
            "recompiled component posterior differs",
        ));
    }
    let scoped = ctx.enter_lowered_dispatch(plan.lowered.identity());
    let child =
        super::cyclic_posterior_action_decision::consume_cyclic_component_posterior_decision(
            &wire.posterior_artifact,
            plan.posterior.lowered().identity(),
            &scoped,
        )?;
    Ok(JointComponentPosteriorDecisionResult {
        child,
        lowered_program_identity: plan.lowered.identity(),
    })
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct VectorComponentOutputWire {
    produced_port: String,
    equation: Equation,
    consumed_ports: Vec<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct VectorComponentWire {
    interface: antecedent_io::interface_contract_artifact::ContractWire,
    outputs: Vec<VectorComponentOutputWire>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct VectorComponentConnectionWire {
    ports: ComponentConnectionWire,
    consumer_output: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct VectorComponentFeedbackWire {
    components: Vec<VectorComponentWire>,
    connections: Vec<VectorComponentConnectionWire>,
    noise_covariance: Vec<f64>,
    noise: Vec<Vec<f64>>,
    initial: Vec<f64>,
    worlds: Vec<ComponentWorldWire>,
    policy: SolvePolicy,
}

impl VectorComponentFeedbackWire {
    fn from_request(request: &VectorFeedbackRequest) -> Self {
        Self {
            components: request
                .components
                .iter()
                .map(|component| VectorComponentWire {
                    interface: antecedent_io::interface_contract_artifact::contract_to_wire(
                        &component.interface,
                    ),
                    outputs: component
                        .outputs
                        .iter()
                        .map(|output| VectorComponentOutputWire {
                            produced_port: output.produced_port.clone(),
                            equation: output.equation.clone(),
                            consumed_ports: output.consumed_ports.clone(),
                        })
                        .collect(),
                })
                .collect(),
            connections: request
                .connections
                .iter()
                .map(|connection| VectorComponentConnectionWire {
                    ports: ComponentConnectionWire {
                        producer: connection.ports.producer.clone(),
                        produced_port: connection.ports.produced_port.clone(),
                        consumer: connection.ports.consumer.clone(),
                        consumed_port: connection.ports.consumed_port.clone(),
                        coefficient: connection.coefficient,
                    },
                    consumer_output: connection.consumer_output.clone(),
                })
                .collect(),
            noise_covariance: request.noise_covariance.clone(),
            noise: request.noise.clone(),
            initial: request.initial.clone(),
            worlds: request
                .worlds
                .iter()
                .map(|world| ComponentWorldWire { clamps: world.clamps.clone() })
                .collect(),
            policy: request.policy,
        }
    }

    fn to_request(&self) -> Result<VectorFeedbackRequest, JointComponentPosteriorError> {
        let components = self
            .components
            .iter()
            .map(|component| {
                let interface = antecedent_io::interface_contract_artifact::contract_from_wire(
                    &component.interface,
                )
                .map_err(|_| {
                    JointComponentPosteriorError::Invalid("invalid vector component interface wire")
                })?;
                Ok(VectorFeedbackComponent {
                    interface,
                    outputs: component
                        .outputs
                        .iter()
                        .map(|output| VectorFeedbackOutput {
                            produced_port: output.produced_port.clone(),
                            equation: output.equation.clone(),
                            consumed_ports: output.consumed_ports.clone(),
                        })
                        .collect(),
                })
            })
            .collect::<Result<Vec<_>, JointComponentPosteriorError>>()?;
        Ok(VectorFeedbackRequest {
            components,
            connections: self
                .connections
                .iter()
                .map(|connection| VectorFeedbackConnection {
                    ports: ComponentConnection::new(
                        connection.ports.producer.as_str(),
                        connection.ports.produced_port.as_str(),
                        connection.ports.consumer.as_str(),
                        connection.ports.consumed_port.as_str(),
                    ),
                    consumer_output: connection.consumer_output.clone(),
                    coefficient: connection.ports.coefficient,
                })
                .collect(),
            noise_covariance: self.noise_covariance.clone(),
            noise: self.noise.clone(),
            initial: self.initial.clone(),
            worlds: self
                .worlds
                .iter()
                .map(|world| ComponentFeedbackWorld { clamps: world.clamps.clone() })
                .collect(),
            policy: self.policy,
            factual: None,
        })
    }
}

/// Checked multi-output component graph and its shared-data joint action program.
#[derive(Clone, Debug)]
pub struct CompiledJointVectorComponentPosteriorDecision {
    component_wire: VectorComponentFeedbackWire,
    schedule: ComponentJointPosteriorSchedule,
    outputs: Vec<super::cyclic_action_world_decision::CyclicDecisionOutput>,
    contract: antecedent_design::decision_contract::DecisionContract,
    posterior: super::cyclic_posterior_action_decision::CompiledCyclicPosteriorDecision,
    lowered: Arc<LoweredProgram>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct JointVectorComponentPosteriorWire {
    version: u32,
    required_feature: String,
    components: VectorComponentFeedbackWire,
    schedule: ComponentJointPosteriorSchedule,
    outputs: Vec<super::cyclic_action_world_decision::CyclicDecisionOutput>,
    contract_artifact: Vec<u8>,
    contract_identity: String,
    program_identity: [u8; 32],
    posterior_artifact: Vec<u8>,
}

const JOINT_VECTOR_COMPONENT_POSTERIOR_FEATURE: &str =
    "joint_vector_component_posterior_decision_v1";

/// Compile checked connected components, one common raw evidence schedule,
/// the joint covariance and action worlds through the cyclic posterior route.
/// The component-preparation noise must be one zero placeholder row; all
/// predictive disturbances are instead generated jointly from its covariance.
///
/// # Errors
/// Invalid component ports, coordinate bindings, evidence or posterior schedule.
pub fn compile_joint_vector_component_posterior_decision(
    request: VectorFeedbackRequest,
    schedule: ComponentJointPosteriorSchedule,
    outputs: Vec<super::cyclic_action_world_decision::CyclicDecisionOutput>,
    contract: antecedent_design::decision_contract::DecisionContract,
    ctx: &ExecutionContext,
) -> Result<CompiledJointVectorComponentPosteriorDecision, JointComponentPosteriorError> {
    if request.factual.is_some()
        || request.noise.len() != 1
        || request.noise[0].iter().any(|value| *value != 0.0)
    {
        return Err(JointComponentPosteriorError::Invalid(
            "posterior handoff requires one zero placeholder noise row and no factual state",
        ));
    }
    if request.components.iter().any(|component| {
        component.interface.produces.iter().any(|port| {
            !matches!(port.support, antecedent_contract::interface_contract::SupportDomain::Real)
        }) || component.interface.consumes.iter().any(|port| {
            !matches!(port.support, antecedent_contract::interface_contract::SupportDomain::Real)
        })
    }) {
        return Err(JointComponentPosteriorError::Invalid(
            "joint posterior handoff requires finite real component ports",
        ));
    }
    let wire = VectorComponentFeedbackWire::from_request(&request);
    let declarations = request.components.clone();
    let prepared = prepare_vector_component_feedback(request, ctx)?;
    for output in &outputs {
        let name = prepared
            .order
            .get(output.equation)
            .ok_or(JointComponentPosteriorError::Invalid("output component port is absent"))?;
        let port = declarations
            .iter()
            .find_map(|component| {
                component.interface.produces.iter().find(|port| {
                    format!("{}::{}", component.interface.component, port.name) == *name
                })
            })
            .ok_or(JointComponentPosteriorError::Invalid(
                "output component port interface is absent",
            ))?;
        let quantity: antecedent_contract::ScientificQuantity =
            output.quantity.clone().try_into().map_err(|_| {
                JointComponentPosteriorError::Invalid("output coordinate is invalid")
            })?;
        let mut declared = port.quantity.clone();
        declared.regime_id = quantity.regime_id.clone();
        declared.require_same_coordinate(&quantity).map_err(|_| {
            JointComponentPosteriorError::Invalid("output coordinate differs from component port")
        })?;
    }
    let cyclic = prepared.prepared.request();
    let posterior_request =
        super::cyclic_posterior_action_decision::CyclicPosteriorDecisionRequest {
            model: cyclic.model.clone(),
            parameters: schedule.parameters.clone(),
            training: schedule.training.clone(),
            prior_sd: schedule.prior_sd,
            chains: schedule.chains,
            warmup: schedule.warmup,
            draws_per_chain: schedule.draws_per_chain,
            proposal_scale: schedule.proposal_scale,
            worlds: cyclic.worlds.clone(),
            initial: cyclic.initial.clone(),
            solve_policy: cyclic.policy,
        };
    let posterior =
        super::cyclic_posterior_action_decision::compile_cyclic_component_posterior_decision(
            posterior_request,
            outputs.clone(),
            contract.clone(),
            ctx,
        )?;
    let component_bytes = antecedent_io::to_cbor(&wire)?;
    let component_digest = CanonicalWriter::digest_bytes(
        "antecedent.joint_vector_component_posterior.component_wire.v1",
        &component_bytes,
    );
    let mut writer =
        CanonicalWriter::new("antecedent.joint_vector_component_posterior.question.v1");
    writer
        .text(JOINT_VECTOR_COMPONENT_POSTERIOR_FEATURE)
        .bytes(&component_digest)
        .bytes(posterior.lowered().identity().as_bytes())
        .bytes(&antecedent_io::to_cbor(&outputs)?);
    let query = writer.finish();
    let mut scientific = posterior.lowered().scientific().clone();
    scientific.target = ProblemTarget::QueryFunctional {
        kind: FunctionalKind::PropagatedChain,
        outcome: None,
        query,
    };
    scientific.plan_id = Arc::from("joint_vector_component_posterior_decision.v1");
    scientific.source_digest = Some(component_digest);
    let identification = IdentificationLayer::NonGraphical(Arc::new(
        NonGraphicalIdentification::SharedEvidenceContract { contract: query },
    ));
    let mut evidence = posterior.lowered().evidence().clone();
    let mut sources = evidence.sources.to_vec();
    sources.push(EvidenceSourceRef {
        role: EvidenceRole::ChainSource,
        digest: Some(component_digest),
    });
    evidence.sources = Arc::from(sources);
    let mut operators = posterior.lowered().physical().operators().to_vec();
    operators.push(
        antecedent_core::PhysicalOperator::declare(antecedent_core::PhysicalEffect::Read)
            .estimates(
                antecedent_core::ByteEstimates::absent().with_output(component_bytes.len() as u64),
            )
            .build()
            .map_err(|_| {
                JointComponentPosteriorError::Invalid("component wire read cannot be lowered")
            })?,
    );
    let physical = PhysicalLayer {
        operators: Arc::from(operators),
        design: posterior.lowered().physical().design(),
    };
    let lowered = Arc::new(
        StandaloneCompositionLoweredParts {
            cell: ProgramCell::JointComponentPosteriorDecision,
            scientific,
            identification,
            statistical: posterior.lowered().statistical().clone(),
            evidence,
            physical,
            claim: *posterior.lowered().claim(),
        }
        .finish(),
    );
    Ok(CompiledJointVectorComponentPosteriorDecision {
        component_wire: wire,
        schedule,
        outputs,
        contract,
        posterior,
        lowered,
    })
}

impl CompiledJointVectorComponentPosteriorDecision {
    /// Outer six-layer program binding the full checked component declaration.
    #[must_use]
    pub fn lowered(&self) -> &LoweredProgram {
        &self.lowered
    }

    /// Inspect the whole compiled route without fitting.
    #[must_use]
    pub fn plan_view(&self) -> PlanView {
        PlanView::of_program(Arc::clone(&self.lowered), PlanStage::Compiled)
    }

    /// Refit the common evidence schedule and solve all paired action worlds.
    /// Connected calibration is unmeasured.
    ///
    /// # Errors
    /// Fit, posterior diagnostics, equilibrium or decision refusal.
    pub fn execute(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<JointComponentPosteriorDecisionResult, JointComponentPosteriorError> {
        let scoped = ctx.enter_lowered_dispatch(self.lowered.identity());
        let child = self.posterior.execute(&scoped)?;
        Ok(JointComponentPosteriorDecisionResult {
            child,
            lowered_program_identity: self.lowered.identity(),
        })
    }

    /// Reexecute and seal both the checked component declarations and joint fit.
    ///
    /// # Errors
    /// A changed result or oversized portable artifact.
    pub fn export(
        &self,
        result: &JointComponentPosteriorDecisionResult,
        ctx: &ExecutionContext,
    ) -> Result<Vec<u8>, JointComponentPosteriorError> {
        if result.lowered_program_identity != self.lowered.identity() {
            return Err(JointComponentPosteriorError::Invalid(
                "component execution program differs",
            ));
        }
        let scoped = ctx.enter_lowered_dispatch(self.lowered.identity());
        let contract = antecedent_design::decision_artifact::DecisionContractArtifact::new(
            self.contract.clone(),
        )
        .map_err(super::cyclic_posterior_action_decision::CyclicPosteriorDecisionError::from)?;
        let bytes = antecedent_io::to_cbor(&JointVectorComponentPosteriorWire {
            version: 1,
            required_feature: JOINT_VECTOR_COMPONENT_POSTERIOR_FEATURE.into(),
            components: self.component_wire.clone(),
            schedule: self.schedule.clone(),
            outputs: self.outputs.clone(),
            contract_identity: contract.identity().into(),
            contract_artifact: contract.to_bytes("joint-component-contract")?,
            program_identity: *self.lowered().identity().as_bytes(),
            posterior_artifact: self.posterior.export(
                &result.child,
                "joint-component-posterior",
                &scoped,
            )?,
        })?;
        if bytes.len() > MAX_JOINT_COMPONENT_BYTES {
            return Err(JointComponentPosteriorError::Invalid(
                "portable component posterior exceeds its bound",
            ));
        }
        Ok(bytes)
    }
}

/// Recheck the component interfaces and coupling, recompile the joint posterior,
/// refit shared raw evidence and independently verify the portable result.
///
/// # Errors
/// Changed feature, component binding, compiler identity, fit or decision.
pub fn consume_joint_vector_component_posterior_decision(
    bytes: &[u8],
    expected_program: SemanticDigest,
    ctx: &ExecutionContext,
) -> Result<JointComponentPosteriorDecisionResult, JointComponentPosteriorError> {
    if bytes.len() > MAX_JOINT_COMPONENT_BYTES {
        return Err(JointComponentPosteriorError::Invalid(
            "portable component posterior exceeds its bound",
        ));
    }
    let wire: JointVectorComponentPosteriorWire = antecedent_io::from_cbor(bytes)?;
    if wire.version != 1
        || wire.required_feature != JOINT_VECTOR_COMPONENT_POSTERIOR_FEATURE
        || wire.program_identity != *expected_program.as_bytes()
    {
        return Err(JointComponentPosteriorError::Invalid(
            "component posterior feature or program differs",
        ));
    }
    let contract = antecedent_design::decision_artifact::DecisionContractArtifact::from_bytes(
        &wire.contract_artifact,
        &wire.contract_identity,
    )?;
    let plan = compile_joint_vector_component_posterior_decision(
        wire.components.to_request()?,
        wire.schedule,
        wire.outputs,
        contract.contract().clone(),
        ctx,
    )?;
    if plan.lowered().identity() != expected_program {
        return Err(JointComponentPosteriorError::Invalid(
            "recompiled component posterior differs",
        ));
    }
    let scoped = ctx.enter_lowered_dispatch(plan.lowered.identity());
    let child =
        super::cyclic_posterior_action_decision::consume_cyclic_component_posterior_decision(
            &wire.posterior_artifact,
            plan.posterior.lowered().identity(),
            &scoped,
        )?;
    Ok(JointComponentPosteriorDecisionResult {
        child,
        lowered_program_identity: plan.lowered.identity(),
    })
}

/// A bounded logistic transform of each real-valued fitted action-world output.
/// This transforms predictive outputs after fitting; the structural ports and
/// Gaussian disturbances remain real-valued and are never claimed bounded.
#[derive(Clone, Debug)]
pub struct CompiledBoundedComponentPosteriorDecision {
    inner: CompiledJointVectorComponentPosteriorDecision,
    quantities: Vec<antecedent_contract::ScientificQuantity>,
    contract: antecedent_design::decision_contract::DecisionContract,
    lowered: Arc<LoweredProgram>,
}

/// Complete fitted producer and bounded predictive action decision.
pub struct BoundedComponentPosteriorDecisionResult {
    /// Original real-valued predictive component fit and paired worlds.
    pub inner: JointComponentPosteriorDecisionResult,
    /// Logistic-transformed paired predictive law and utility.
    pub decision: super::propagation_chain_decision::ChainDecisionResult,
    /// Full compiler identity.
    pub lowered_program_identity: SemanticDigest,
    /// Always unmeasured.
    pub calibration: &'static str,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct BoundedComponentPosteriorWire {
    version: u32,
    feature: String,
    program_identity: [u8; 32],
    inner_identity: [u8; 32],
    inner_artifact: Vec<u8>,
    quantities: Vec<antecedent_io::quantity_wire::ScientificQuantityWire>,
    contract_identity: String,
    contract_artifact: Vec<u8>,
    chain_identity: [u8; 32],
    chain_artifact: Vec<u8>,
}

/// Compile logistic bounded outputs after the real-valued joint component fit.
/// Each output retains the source variable, regime, population and functional;
/// its units must be dimensionless and transform identity `logistic`.
///
/// # Errors
/// Wrong output coordinates, action contract, or unsupported lowering.
pub fn compile_bounded_component_posterior_decision(
    inner: CompiledJointVectorComponentPosteriorDecision,
    quantities: Vec<antecedent_contract::ScientificQuantity>,
    contract: antecedent_design::decision_contract::DecisionContract,
) -> Result<CompiledBoundedComponentPosteriorDecision, JointComponentPosteriorError> {
    contract
        .validate()
        .map_err(super::cyclic_posterior_action_decision::CyclicPosteriorDecisionError::from)?;
    if quantities.len() != inner.outputs.len() || quantities.len() < 2 {
        return Err(JointComponentPosteriorError::Invalid(
            "bounded outputs must cover every paired component world",
        ));
    }
    for (quantity, output) in quantities.iter().zip(&inner.outputs) {
        let raw: antecedent_contract::ScientificQuantity =
            output.quantity.clone().try_into().map_err(|_| {
                JointComponentPosteriorError::Invalid("invalid raw component quantity")
            })?;
        let mut expected = raw.clone();
        expected.units = "dimensionless".into();
        expected.transform_id = "logistic".into();
        if expected.require_same_coordinate(quantity).is_err()
            || quantity.population_id != contract.target_population
            || quantity.horizon != contract.horizon
        {
            return Err(JointComponentPosteriorError::Invalid(
                "bounded output must be the logistic transform of its exact real port",
            ));
        }
    }
    if contract.actions.len() != quantities.len()
        || contract.actions.iter().any(|action| {
            action.inputs.len() != 1
                || !quantities
                    .iter()
                    .any(|quantity| action.inputs[0].require_same_coordinate(quantity).is_ok())
        })
    {
        return Err(JointComponentPosteriorError::Invalid(
            "bounded actions must each name an output world",
        ));
    }
    let quantity_wires: Vec<_> =
        quantities.iter().map(antecedent_io::quantity_wire::ScientificQuantityWire::from).collect();
    let mut writer = CanonicalWriter::new("antecedent.bounded_component_posterior.question.v1");
    writer
        .bytes(inner.lowered.identity().as_bytes())
        .bytes(&antecedent_io::to_cbor(&quantity_wires)?)
        .text(
            &contract.identity().map_err(
                super::cyclic_posterior_action_decision::CyclicPosteriorDecisionError::from,
            )?,
        )
        .text("logistic_after_joint_predictive_fit_v1");
    let query = writer.finish();
    let mut scientific = inner.lowered.scientific().clone();
    scientific.target =
        ProblemTarget::QueryFunctional { kind: FunctionalKind::PolicyValue, outcome: None, query };
    scientific.plan_id = Arc::from("bounded_joint_component_posterior_decision.v1");
    scientific.graph_class = "real_fitted_component_logistic_predictive_output";
    let identification = IdentificationLayer::NonGraphical(Arc::new(
        NonGraphicalIdentification::SharedEvidenceContract { contract: query },
    ));
    let mut statistical = inner.lowered.statistical().clone();
    statistical.realization = crate::planner::statistical::Realization::NestedFit {
        producer_program: inner.lowered.identity(),
        method: Arc::from("joint_component_posterior_logistic_predictive_v1"),
        law: crate::planner::statistical::NestedFitLaw::Predictive,
    };
    let mut evidence = inner.lowered.evidence().clone();
    let mut sources = evidence.sources.to_vec();
    sources.push(EvidenceSourceRef {
        role: EvidenceRole::ChainSource,
        digest: Some(*inner.lowered.identity().as_bytes()),
    });
    evidence.sources = Arc::from(sources);
    let mut operators = inner.lowered.physical().operators().to_vec();
    operators.push(
        antecedent_core::PhysicalOperator::declare(antecedent_core::PhysicalEffect::Condition)
            .build()
            .map_err(|_| {
                JointComponentPosteriorError::Invalid("bounded transform cannot be lowered")
            })?,
    );
    let lowered = Arc::new(
        StandaloneCompositionLoweredParts {
            cell: ProgramCell::BoundedComponentPosteriorDecision,
            scientific,
            identification,
            statistical,
            evidence,
            physical: PhysicalLayer {
                operators: Arc::from(operators),
                design: inner.lowered.physical().design(),
            },
            claim: *inner.lowered.claim(),
        }
        .finish(),
    );
    Ok(CompiledBoundedComponentPosteriorDecision { inner, quantities, contract, lowered })
}

impl CompiledBoundedComponentPosteriorDecision {
    /// Full component, fit, transform and decision compiler program.
    #[must_use]
    pub fn lowered(&self) -> &LoweredProgram {
        &self.lowered
    }
    /// Inspect the retained program before fitting.
    #[must_use]
    pub fn plan_view(&self) -> PlanView {
        PlanView::of_program(Arc::clone(&self.lowered), PlanStage::Compiled)
    }

    fn chain(
        &self,
        inner: &JointComponentPosteriorDecisionResult,
        ctx: &ExecutionContext,
    ) -> Result<super::propagation_chain_decision::ChainDecisionPlan, JointComponentPosteriorError>
    {
        use antecedent_contract::dependence::SharedFactorKind;
        use antecedent_contract::propagation_chain::{
            ChainReading, LawMeaning, StageKind, TableConstruction,
        };
        use antecedent_io::propagation_chain_record::{
            ChainJointOutputWire, ChainOpWire, ChainSourceWire, ChainStageWire, ChainUnaryWire,
            PropagationChainRecordWire,
        };
        let (columns, weights) =
            inner.child.decision.chain.joint_output_law().ok_or(
                JointComponentPosteriorError::Invalid("real fitted worlds have no joint law"),
            )?;
        if columns.len() != self.quantities.len() || weights.is_some() {
            return Err(JointComponentPosteriorError::Invalid(
                "bounded route requires unweighted aligned fitted worlds",
            ));
        }
        let alignment = inner.child.decision.chain.identity();
        let mut sources = Vec::new();
        let mut stages = Vec::new();
        let mut outputs = Vec::new();
        for (i, (column, quantity)) in columns.iter().zip(&self.quantities).enumerate() {
            let source = format!("real.world.{i}");
            let stage = format!("bounded.world.{i}");
            let values: Vec<_> = column.iter().copied().map(f64::from_bits).collect();
            sources.push(
                ChainSourceWire::draws(&source, LawMeaning::Predictive, alignment, &values)
                    .with_factor(SharedFactorKind::Model, alignment),
            );
            stages.push(ChainStageWire::new(
                &stage,
                StageKind::DerivedQuantity,
                &[&source],
                ChainOpWire::unary(ChainUnaryWire::Logistic),
            ));
            outputs.push(ChainJointOutputWire {
                stage,
                quantity: antecedent_io::quantity_wire::ScientificQuantityWire::from(quantity),
            });
        }
        let draft = PropagationChainRecordWire::draft(
            TableConstruction::JointDraws,
            ChainReading::Predictive,
            &outputs[0].stage,
            sources,
            stages,
        )
        .with_output_quantity(outputs[0].quantity.clone())
        .with_joint_outputs(outputs);
        super::propagation_chain_decision::compile_action_world_chain_decision(
            draft,
            self.contract.clone(),
            ctx,
        )
        .map_err(super::cyclic_posterior_action_decision::CyclicPosteriorDecisionError::from)
        .map_err(Into::into)
    }

    /// Refit all components and evaluate logistic outputs under their shared worlds.
    ///
    /// # Errors
    /// A fit, diagnostics, equilibrium, transform or decision refusal.
    pub fn execute(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<BoundedComponentPosteriorDecisionResult, JointComponentPosteriorError> {
        let scoped = ctx.enter_lowered_dispatch(self.lowered.identity());
        let inner = self.inner.execute(&scoped)?;
        let decision = self
            .chain(&inner, &scoped)?
            .execute(&scoped)
            .map_err(super::cyclic_posterior_action_decision::CyclicPosteriorDecisionError::from)?;
        Ok(BoundedComponentPosteriorDecisionResult {
            inner,
            decision,
            lowered_program_identity: self.lowered.identity(),
            calibration: "not_measured",
        })
    }

    /// Seal a fresh complete fit and transformed decision for replay.
    ///
    /// # Errors
    /// Changed fit/decision or oversized artifact.
    pub fn export(
        &self,
        result: &BoundedComponentPosteriorDecisionResult,
        ctx: &ExecutionContext,
    ) -> Result<Vec<u8>, JointComponentPosteriorError> {
        let live = self.execute(ctx)?;
        if result.lowered_program_identity != self.lowered.identity()
            || result.calibration != "not_measured"
            || result.inner.lowered_program_identity != live.inner.lowered_program_identity
            || result.inner.child.lowered_program_identity
                != live.inner.child.lowered_program_identity
            || result.inner.child.posterior != live.inner.child.posterior
            || result.inner.child.decision.decision != live.inner.child.decision.decision
            || result.inner.child.decision.chain.identity()
                != live.inner.child.decision.chain.identity()
            || result.decision.chain.identity() != live.decision.chain.identity()
            || result.decision.decision != live.decision.decision
        {
            return Err(JointComponentPosteriorError::Invalid(
                "bounded component result differs on replay",
            ));
        }
        let contract = antecedent_design::decision_artifact::DecisionContractArtifact::new(
            self.contract.clone(),
        )
        .map_err(super::cyclic_posterior_action_decision::CyclicPosteriorDecisionError::from)?;
        let chain = self.chain(&live.inner, ctx)?;
        let bytes = antecedent_io::to_cbor(&BoundedComponentPosteriorWire {
            version: 1,
            feature: "bounded_joint_component_posterior_decision_v1".into(),
            program_identity: *self.lowered.identity().as_bytes(),
            inner_identity: *self.inner.lowered.identity().as_bytes(),
            inner_artifact: self.inner.export(&live.inner, ctx)?,
            quantities: self
                .quantities
                .iter()
                .map(antecedent_io::quantity_wire::ScientificQuantityWire::from)
                .collect(),
            contract_identity: contract.identity().into(),
            contract_artifact: contract.to_bytes("bounded-component-contract")?,
            chain_identity: *chain.lowered().identity().as_bytes(),
            chain_artifact: chain
                .export(&live.decision, "bounded-component-chain")
                .map_err(|error| JointComponentPosteriorError::Posterior(error.into()))?,
        })?;
        if bytes.len() > MAX_JOINT_COMPONENT_BYTES {
            return Err(JointComponentPosteriorError::Invalid(
                "bounded component artifact exceeds its bound",
            ));
        }
        Ok(bytes)
    }
}

/// Independently recompile the vector component source, refit and replay the
/// logistic action-world chain under an externally retained program identity.
///
/// # Errors
/// Changed source, fit, transform, contract or decision.
pub fn consume_bounded_component_posterior_decision(
    bytes: &[u8],
    expected: SemanticDigest,
    ctx: &ExecutionContext,
) -> Result<BoundedComponentPosteriorDecisionResult, JointComponentPosteriorError> {
    if bytes.len() > MAX_JOINT_COMPONENT_BYTES {
        return Err(JointComponentPosteriorError::Invalid(
            "bounded component artifact exceeds its bound",
        ));
    }
    let wire: BoundedComponentPosteriorWire = antecedent_io::from_cbor(bytes)?;
    if wire.version != 1
        || wire.feature != "bounded_joint_component_posterior_decision_v1"
        || wire.program_identity != *expected.as_bytes()
    {
        return Err(JointComponentPosteriorError::Invalid(
            "bounded component feature or identity differs",
        ));
    }
    let retained = consume_joint_vector_component_posterior_decision(
        &wire.inner_artifact,
        SemanticDigest::from_bytes(wire.inner_identity),
        ctx,
    )?;
    let inner_wire: JointVectorComponentPosteriorWire =
        antecedent_io::from_cbor(&wire.inner_artifact)?;
    let inner_contract =
        antecedent_design::decision_artifact::DecisionContractArtifact::from_bytes(
            &inner_wire.contract_artifact,
            &inner_wire.contract_identity,
        )?;
    let inner = compile_joint_vector_component_posterior_decision(
        inner_wire.components.to_request()?,
        inner_wire.schedule,
        inner_wire.outputs,
        inner_contract.contract().clone(),
        ctx,
    )?;
    let contract = antecedent_design::decision_artifact::DecisionContractArtifact::from_bytes(
        &wire.contract_artifact,
        &wire.contract_identity,
    )?;
    let quantities = wire
        .quantities
        .into_iter()
        .map(|q| {
            q.try_into().map_err(|_| {
                JointComponentPosteriorError::Invalid("bounded output coordinate is invalid")
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let plan = compile_bounded_component_posterior_decision(
        inner,
        quantities,
        contract.contract().clone(),
    )?;
    if plan.lowered.identity() != expected {
        return Err(JointComponentPosteriorError::Invalid("bounded component compiler differs"));
    }
    let live = plan.execute(ctx)?;
    if live.inner.lowered_program_identity != retained.lowered_program_identity
        || live.inner.child.lowered_program_identity != retained.child.lowered_program_identity
        || live.inner.child.posterior != retained.child.posterior
        || live.inner.child.decision.chain.identity() != retained.child.decision.chain.identity()
        || live.inner.child.decision.decision != retained.child.decision.decision
        || live.decision.lowered_program_identity != SemanticDigest::from_bytes(wire.chain_identity)
    {
        return Err(JointComponentPosteriorError::Invalid(
            "bounded component inner result differs",
        ));
    }
    let replay = super::propagation_chain_decision::consume_chain_decision(
        &wire.chain_artifact,
        Some(live.decision.lowered_program_identity),
        ctx,
    )
    .map_err(super::cyclic_posterior_action_decision::CyclicPosteriorDecisionError::from)?;
    if replay.chain.identity() != live.decision.chain.identity()
        || replay.decision != live.decision.decision
    {
        return Err(JointComponentPosteriorError::Invalid("bounded component decision differs"));
    }
    Ok(live)
}

/// Prospective reliability scope for the full bounded-output decision.
pub struct BoundedComponentReliabilityCandidate {
    /// Complete fitted and transformed decision.
    pub result: BoundedComponentPosteriorDecisionResult,
    /// Exact source-bound design identity for a future reliability record.
    pub scope: SemanticDigest,
    /// Always `not_measured`.
    pub calibration: &'static str,
}

impl CompiledBoundedComponentPosteriorDecision {
    /// Fixed, pre-fit reliability scope excluding observed training values and seed.
    ///
    /// # Errors
    /// Either inner or outer criterion is not eligible, constraints are present,
    /// or the fixed declaration cannot be encoded.
    pub fn reliability_scope(&self) -> Result<SemanticDigest, JointComponentPosteriorError> {
        use antecedent_design::decision_contract::DecisionCriterion;
        if !matches!(
            self.contract.criterion,
            DecisionCriterion::PosteriorExpectedUtility | DecisionCriterion::ExpectedRegret
        ) || !self.contract.constraints.is_empty()
        {
            return Err(JointComponentPosteriorError::Invalid(
                "bounded reliability needs unconstrained expected utility or expected regret",
            ));
        }
        let inner_scope = self.inner.posterior.decision_reliability_scope()?;
        let component_bytes = antecedent_io::to_cbor(&self.inner.component_wire)?;
        let component_digest = CanonicalWriter::digest_bytes(
            "antecedent.bounded_component.reliability_component.v1",
            &component_bytes,
        );
        let quantities: Vec<_> = self
            .quantities
            .iter()
            .map(antecedent_io::quantity_wire::ScientificQuantityWire::from)
            .collect();
        let mut writer = CanonicalWriter::new("antecedent.bounded_component.reliability_scope.v1");
        writer
            .bytes(inner_scope.as_bytes())
            .bytes(&component_digest)
            .bytes(&antecedent_io::to_cbor(&quantities)?)
            .text(&self.contract.identity().map_err(
                super::cyclic_posterior_action_decision::CyclicPosteriorDecisionError::from,
            )?)
            .text("logistic_after_joint_predictive_fit_v1");
        Ok(SemanticDigest::from_bytes(writer.finish()))
    }

    /// Execute the full route and expose its pre-fit, unmeasured reliability scope.
    ///
    /// # Errors
    /// Ineligible scope or any nested fit or decision refusal.
    pub fn reliability_candidate(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<BoundedComponentReliabilityCandidate, JointComponentPosteriorError> {
        let scope = self.reliability_scope()?;
        let result = self.execute(ctx)?;
        Ok(BoundedComponentReliabilityCandidate { result, scope, calibration: "not_measured" })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_contract::evidence_ledger::{ObservationId, SourceDeclaration, SourceKind};
    use antecedent_contract::interface_contract::{
        ConsumedPort, ContractOperation, ProducedPort, SupportDomain,
    };
    use antecedent_contract::{QuantityRole, ScientificQuantity};
    use antecedent_model::nonlinear_equilibrium::Activation;

    fn quantity() -> ScientificQuantity {
        ScientificQuantity {
            variable_id: "game:wishlists".into(),
            variable_name: "wishlists".into(),
            role: QuantityRole::Outcome,
            units: "count".into(),
            population_id: "game-a".into(),
            regime_id: "specified-equilibrium".into(),
            horizon: 1,
            functional_id: "equilibrium_value".into(),
            conditioning: Vec::new(),
            transform_id: "identity".into(),
        }
    }

    fn component(name: &str, self_weight: f64) -> ScalarFeedbackComponent {
        ScalarFeedbackComponent {
            interface: InterfaceContract {
                component: name.into(),
                produces: vec![ProducedPort::new(
                    "out",
                    quantity(),
                    "model:specified",
                    SupportDomain::Real,
                    [ContractOperation::Mean],
                )],
                consumes: vec![ConsumedPort::new(
                    "in",
                    quantity(),
                    "model:specified",
                    SupportDomain::Real,
                    [ContractOperation::Mean],
                )],
            },
            equation: Equation {
                id: format!("{name}.mechanism"),
                intercept: 1.0,
                weights: vec![self_weight],
                activation: Activation::Identity,
                expression: None,
                parameters: Vec::new(),
            },
        }
    }

    fn request() -> ComponentFeedbackRequest {
        ComponentFeedbackRequest {
            components: vec![component("wishlists", 0.2), component("traffic", 0.1)],
            connections: vec![
                FeedbackConnection {
                    ports: ComponentConnection::new("traffic", "out", "wishlists", "in"),
                    coefficient: 0.3,
                },
                FeedbackConnection {
                    ports: ComponentConnection::new("wishlists", "out", "traffic", "in"),
                    coefficient: 0.2,
                },
            ],
            noise_covariance: vec![1.0, 0.0, 0.0, 1.0],
            noise: vec![vec![0.0, 0.0]],
            initial: vec![0.0, 0.0],
            worlds: vec![ComponentFeedbackWorld { clamps: Vec::new() }],
            policy: SolvePolicy::default(),
            factual: None,
        }
    }

    #[test]
    #[ignore = "source-only joint component posterior; no connected calibration is earned"]
    fn common_training_rows_refit_and_replay_the_connected_action_decision() {
        use super::super::cyclic_action_world_decision::CyclicDecisionOutput;
        use antecedent_design::decision_contract::{
            ActionKind, DecisionAction, DecisionContract, DecisionCriterion, StructuralPolicy,
            UtilityExpr,
        };
        use antecedent_estimate::cyclic_likelihood::CyclicParameter;
        use antecedent_io::quantity_wire::ScientificQuantityWire;
        use antecedent_model::nonlinear_equilibrium::RegimeRow;
        let mut components = request();
        components.noise_covariance = vec![0.04, 0.0, 0.0, 0.04];
        components.worlds = vec![
            ComponentFeedbackWorld { clamps: vec![("traffic".into(), 0.0)] },
            ComponentFeedbackWorld { clamps: vec![("traffic".into(), 1.0)] },
        ];
        let training = (0..260)
            .map(|i| {
                let traffic = (i % 5) as f64 * 0.4;
                let disturbance = (i as f64 * 0.19).sin() * 0.2;
                RegimeRow {
                    values: vec![traffic, (1.0 + 0.3 * traffic + disturbance) / 0.8],
                    clamps: vec![Clamp { variable: 0, value: traffic }],
                }
            })
            .collect();
        let schedule = ComponentJointPosteriorSchedule {
            parameters: vec![CyclicParameter::Weight { equation: 1, parent: 0 }],
            training,
            prior_sd: 1.0,
            chains: 2,
            warmup: 300,
            draws_per_chain: 300,
            proposal_scale: 0.08,
        };
        let quantities: Vec<_> = ["ordinary", "promoted"]
            .iter()
            .map(|regime| {
                let mut q = quantity();
                q.regime_id = (*regime).into();
                q
            })
            .collect();
        let outputs = quantities
            .iter()
            .enumerate()
            .map(|(world, quantity)| CyclicDecisionOutput {
                world,
                equation: 1,
                quantity: ScientificQuantityWire::from(quantity),
            })
            .collect();
        let contract = DecisionContract {
            actions: quantities
                .iter()
                .enumerate()
                .map(|(i, q)| DecisionAction {
                    id: ["ordinary", "promoted"][i].into(),
                    kind: ActionKind::Intervention,
                    inputs: vec![q.clone()],
                    utility: UtilityExpr::Input(0),
                })
                .collect(),
            utility_units: "count".into(),
            criterion: DecisionCriterion::ExpectedRegret,
            constraints: vec![],
            target_population: quantities[0].population_id.clone(),
            horizon: quantities[0].horizon,
            structural_policy: StructuralPolicy::RequireInvariantBestAction,
        };
        let ctx = ExecutionContext::for_tests(47);
        let plan = compile_joint_component_posterior_decision(
            components, schedule, outputs, contract, &ctx,
        )
        .expect("compile checked components and common fit");
        let result = plan.execute(&ctx).expect("joint posterior and action worlds");
        let bytes = plan.export(&result, &ctx).expect("portable handoff");
        let replay =
            consume_joint_component_posterior_decision(&bytes, plan.lowered().identity(), &ctx)
                .expect("independent component and fit replay");
        assert_eq!(replay.child.decision.decision, result.child.decision.decision);
        assert_eq!(replay.lowered_program_identity, plan.lowered().identity());
        assert_eq!(plan.plan_view().route().as_str(), "JointComponentPosteriorDecision");
        let mut reordered: JointComponentPosteriorWire = antecedent_io::from_cbor(&bytes).unwrap();
        reordered.components.components.reverse();
        let changed_plan = compile_joint_component_posterior_decision(
            reordered.components.to_request().unwrap(),
            reordered.schedule.clone(),
            reordered.outputs.clone(),
            plan.contract.clone(),
            &ctx,
        )
        .unwrap();
        assert_eq!(
            changed_plan.posterior.lowered().identity(),
            plan.posterior.lowered().identity()
        );
        assert_ne!(changed_plan.lowered().identity(), plan.lowered().identity());
        let changed = antecedent_io::to_cbor(&reordered).unwrap();
        assert!(
            consume_joint_component_posterior_decision(&changed, plan.lowered().identity(), &ctx,)
                .is_err()
        );
        let mut wire: JointComponentPosteriorWire = antecedent_io::from_cbor(&bytes).unwrap();
        wire.outputs[0].quantity.variable_id = "different-port".into();
        let changed = antecedent_io::to_cbor(&wire).unwrap();
        assert!(
            consume_joint_component_posterior_decision(&changed, plan.lowered().identity(), &ctx,)
                .is_err()
        );
    }

    #[test]
    fn cross_component_cycle_is_one_compiler_program() {
        let ctx = ExecutionContext::for_tests(37);
        let prepared = prepare_component_feedback(request(), &ctx).expect("connected compiler");
        assert_eq!(prepared.component_order(), &["traffic".to_owned(), "wishlists".to_owned()]);
        let equations = &prepared.prepared().request().model.equations;
        assert_eq!(equations[0].weights, [0.1, 0.2]);
        assert_eq!(equations[1].weights, [0.3, 0.2]);
        assert!(prepared.prepared().lowered().identity().as_bytes().iter().any(|b| *b != 0));
    }

    #[test]
    fn feedback_requires_global_contraction_semantics() {
        let ctx = ExecutionContext::for_tests(37);
        let mut draft = request();
        draft.policy.require_contraction = false;
        assert!(prepare_component_feedback(draft, &ctx).is_err());
    }

    #[test]
    fn aggregate_ledger_checks_disjoint_inputs_and_binds_compiler_identity() {
        let ctx = ExecutionContext::for_tests(37);
        let mut ledger = EvidenceLedger::new();
        let first = ledger
            .add_source(
                SourceDeclaration::new("ads", SourceKind::ObservationSet)
                    .with_observations([ObservationId::from_key("campaign", b"ads")]),
            )
            .expect("first source");
        let second = ledger
            .add_source(
                SourceDeclaration::new("community", SourceKind::ObservationSet)
                    .with_observations([ObservationId::from_key("campaign", b"community")]),
            )
            .expect("second source");
        let mut draft = request();
        draft.components.push(component("community", 0.1));
        draft.components[0].interface.consumes.push(
            ConsumedPort::new(
                "community_in",
                quantity(),
                "model:specified",
                SupportDomain::Real,
                [ContractOperation::Mean],
            )
            .with_dependence(DependenceRequirement::DisjointEvidence),
        );
        draft.components[0].interface.consumes[0].dependence =
            DependenceRequirement::DisjointEvidence;
        draft.components[1].interface.produces[0].evidence =
            EvidenceBasis::Declared([first].into());
        draft.components[2].interface.produces[0].evidence =
            EvidenceBasis::Declared([second].into());
        draft.connections.push(FeedbackConnection {
            ports: ComponentConnection::new("community", "out", "wishlists", "community_in"),
            coefficient: 0.1,
        });
        draft.connections.push(FeedbackConnection {
            ports: ComponentConnection::new("wishlists", "out", "community", "in"),
            coefficient: 0.1,
        });
        draft.noise_covariance = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        draft.noise = vec![vec![0.0; 3]];
        draft.initial = vec![0.0; 3];
        assert!(prepare_component_feedback(draft.clone(), &ctx).is_err());
        let prepared = prepare_component_feedback_with_ledger(draft.clone(), &ledger, &ctx)
            .expect("known disjoint evidence");
        assert_eq!(prepared.component_order().len(), 3);
        let mut changed_ledger = ledger.clone();
        changed_ledger
            .add_source(
                SourceDeclaration::new("later", SourceKind::ObservationSet)
                    .with_observations([ObservationId::from_key("campaign", b"later")]),
            )
            .expect("later source");
        let changed = prepare_component_feedback_with_ledger(draft.clone(), &changed_ledger, &ctx)
            .expect("changed ledger still proves disjointness");
        assert_ne!(
            prepared.prepared().lowered().identity(),
            changed.prepared().lowered().identity()
        );
        draft.components[2].interface.produces[0].evidence =
            EvidenceBasis::Declared([first].into());
        assert!(prepare_component_feedback_with_ledger(draft, &ledger, &ctx).is_err());
    }

    #[test]
    fn multi_output_component_lowers_one_connected_cyclic_program() {
        let original = request();
        let mut traffic = original.components[1].clone();
        traffic.interface.produces[0].name = "traffic".into();
        traffic.interface.produces[0].quantity.variable_id = "game:traffic".into();
        traffic.interface.produces[0].quantity.variable_name = "traffic".into();
        traffic.interface.consumes[0].name = "from_wishlists".into();
        let mut wishlists = original.components[0].clone();
        wishlists.interface.produces[0].name = "wishlists".into();
        wishlists.interface.consumes[0].name = "from_traffic".into();
        wishlists.interface.consumes[0].quantity.variable_id = "game:traffic".into();
        wishlists.interface.consumes[0].quantity.variable_name = "traffic".into();
        let vector = VectorFeedbackRequest {
            components: vec![VectorFeedbackComponent {
                interface: InterfaceContract {
                    component: "game".into(),
                    produces: vec![
                        traffic.interface.produces[0].clone(),
                        wishlists.interface.produces[0].clone(),
                    ],
                    consumes: vec![
                        traffic.interface.consumes[0].clone(),
                        wishlists.interface.consumes[0].clone(),
                    ],
                },
                outputs: vec![
                    VectorFeedbackOutput {
                        produced_port: "traffic".into(),
                        equation: traffic.equation,
                        consumed_ports: vec!["from_wishlists".into()],
                    },
                    VectorFeedbackOutput {
                        produced_port: "wishlists".into(),
                        equation: wishlists.equation,
                        consumed_ports: vec!["from_traffic".into()],
                    },
                ],
            }],
            connections: vec![
                VectorFeedbackConnection {
                    ports: ComponentConnection::new("game", "traffic", "game", "from_traffic"),
                    consumer_output: "wishlists".into(),
                    coefficient: 0.3,
                },
                VectorFeedbackConnection {
                    ports: ComponentConnection::new("game", "wishlists", "game", "from_wishlists"),
                    consumer_output: "traffic".into(),
                    coefficient: 0.2,
                },
            ],
            noise_covariance: original.noise_covariance,
            noise: original.noise,
            initial: original.initial,
            worlds: original.worlds,
            policy: original.policy,
            factual: original.factual,
        };
        let ctx = ExecutionContext::for_tests(37);
        let prepared =
            prepare_vector_component_feedback(vector, &ctx).expect("multi-output cyclic compiler");
        assert_eq!(
            prepared.component_order(),
            &["game::traffic".to_owned(), "game::wishlists".to_owned(),]
        );
        let equations = &prepared.prepared().request().model.equations;
        assert_eq!(equations[0].weights, [0.1, 0.2]);
        assert_eq!(equations[1].weights, [0.3, 0.2]);
    }
    #[test]
    #[ignore = "source-only joint vector component posterior; connected calibration is not earned"]
    fn vector_components_fit_joint_worlds_and_replay() {
        use super::super::cyclic_action_world_decision::CyclicDecisionOutput;
        use antecedent_design::decision_contract::{
            ActionKind, DecisionAction, DecisionContract, DecisionCriterion, StructuralPolicy,
            UtilityExpr,
        };
        use antecedent_estimate::cyclic_likelihood::CyclicParameter;
        use antecedent_io::quantity_wire::ScientificQuantityWire;
        use antecedent_model::nonlinear_equilibrium::RegimeRow;
        let original = request();
        let mut traffic = original.components[1].clone();
        traffic.interface.produces[0].name = "traffic".into();
        traffic.interface.produces[0].quantity.variable_id = "game:traffic".into();
        traffic.interface.produces[0].quantity.variable_name = "traffic".into();
        traffic.interface.consumes[0].name = "from_wishlists".into();
        let mut wishlists = original.components[0].clone();
        wishlists.interface.produces[0].name = "wishlists".into();
        wishlists.interface.consumes[0].name = "from_traffic".into();
        wishlists.interface.consumes[0].quantity.variable_id = "game:traffic".into();
        wishlists.interface.consumes[0].quantity.variable_name = "traffic".into();
        let mut vector = VectorFeedbackRequest {
            components: vec![VectorFeedbackComponent {
                interface: InterfaceContract {
                    component: "game".into(),
                    produces: vec![
                        traffic.interface.produces[0].clone(),
                        wishlists.interface.produces[0].clone(),
                    ],
                    consumes: vec![
                        traffic.interface.consumes[0].clone(),
                        wishlists.interface.consumes[0].clone(),
                    ],
                },
                outputs: vec![
                    VectorFeedbackOutput {
                        produced_port: "traffic".into(),
                        equation: traffic.equation,
                        consumed_ports: vec!["from_wishlists".into()],
                    },
                    VectorFeedbackOutput {
                        produced_port: "wishlists".into(),
                        equation: wishlists.equation,
                        consumed_ports: vec!["from_traffic".into()],
                    },
                ],
            }],
            connections: vec![
                VectorFeedbackConnection {
                    ports: ComponentConnection::new("game", "traffic", "game", "from_traffic"),
                    consumer_output: "wishlists".into(),
                    coefficient: 0.3,
                },
                VectorFeedbackConnection {
                    ports: ComponentConnection::new("game", "wishlists", "game", "from_wishlists"),
                    consumer_output: "traffic".into(),
                    coefficient: 0.2,
                },
            ],
            noise_covariance: original.noise_covariance,
            noise: original.noise,
            initial: original.initial,
            worlds: original.worlds,
            policy: original.policy,
            factual: original.factual,
        };
        vector.noise_covariance = vec![0.04, 0.0, 0.0, 0.04];
        vector.worlds = vec![
            ComponentFeedbackWorld { clamps: vec![("game::traffic".into(), 0.0)] },
            ComponentFeedbackWorld { clamps: vec![("game::traffic".into(), 1.0)] },
        ];
        let training = (0..260)
            .map(|i| {
                let traffic = (i % 5) as f64 * 0.4;
                let disturbance = (i as f64 * 0.19).sin() * 0.2;
                RegimeRow {
                    values: vec![traffic, (1.0 + 0.3 * traffic + disturbance) / 0.8],
                    clamps: vec![Clamp { variable: 0, value: traffic }],
                }
            })
            .collect();
        let schedule = ComponentJointPosteriorSchedule {
            parameters: vec![CyclicParameter::Weight { equation: 1, parent: 0 }],
            training,
            prior_sd: 1.0,
            chains: 2,
            warmup: 300,
            draws_per_chain: 300,
            proposal_scale: 0.08,
        };
        let quantities: Vec<_> = ["ordinary", "promoted"]
            .iter()
            .map(|regime| {
                let mut q = quantity();
                q.regime_id = (*regime).into();
                q
            })
            .collect();
        let outputs: Vec<_> = quantities
            .iter()
            .enumerate()
            .map(|(world, quantity)| CyclicDecisionOutput {
                world,
                equation: 1,
                quantity: ScientificQuantityWire::from(quantity),
            })
            .collect();
        let contract = DecisionContract {
            actions: quantities
                .iter()
                .enumerate()
                .map(|(i, q)| DecisionAction {
                    id: ["ordinary", "promoted"][i].into(),
                    kind: ActionKind::Intervention,
                    inputs: vec![q.clone()],
                    utility: UtilityExpr::Input(0),
                })
                .collect(),
            utility_units: "count".into(),
            criterion: DecisionCriterion::ExpectedRegret,
            constraints: vec![],
            target_population: quantities[0].population_id.clone(),
            horizon: quantities[0].horizon,
            structural_policy: StructuralPolicy::RequireInvariantBestAction,
        };
        let ctx = ExecutionContext::for_tests(47);
        let plan = compile_joint_vector_component_posterior_decision(
            vector.clone(),
            schedule.clone(),
            outputs.clone(),
            contract.clone(),
            &ctx,
        )
        .unwrap();
        let result = plan.execute(&ctx).unwrap();
        let artifact = plan.export(&result, &ctx).unwrap();
        let replay = consume_joint_vector_component_posterior_decision(
            &artifact,
            plan.lowered().identity(),
            &ctx,
        )
        .unwrap();
        assert_eq!(replay.child.decision.decision, result.child.decision.decision);
        let mut invalid = vector.clone();
        invalid.connections[0].consumer_output = "missing".into();
        assert!(
            compile_joint_vector_component_posterior_decision(
                invalid,
                schedule.clone(),
                outputs.clone(),
                contract.clone(),
                &ctx
            )
            .is_err()
        );
        let mut invalid = vector;
        invalid.components[0].interface.produces[0].support = SupportDomain::Binary;
        assert!(
            compile_joint_vector_component_posterior_decision(
                invalid, schedule, outputs, contract, &ctx
            )
            .is_err()
        );
        let mut wire: JointVectorComponentPosteriorWire =
            antecedent_io::from_cbor(&artifact).unwrap();
        wire.components.connections[0].consumer_output = "missing".into();
        let changed = antecedent_io::to_cbor(&wire).unwrap();
        assert!(
            consume_joint_vector_component_posterior_decision(
                &changed,
                plan.lowered().identity(),
                &ctx
            )
            .is_err()
        );
    }

    #[test]
    #[ignore = "source-only bounded fitted component output; reliability not earned"]
    fn bounded_component_worlds_replay_after_joint_fit() {
        use super::super::cyclic_action_world_decision::CyclicDecisionOutput;
        use antecedent_design::decision_contract::{
            ActionKind, DecisionAction, DecisionContract, DecisionCriterion, StructuralPolicy,
            UtilityExpr,
        };
        use antecedent_estimate::cyclic_likelihood::CyclicParameter;
        use antecedent_io::quantity_wire::ScientificQuantityWire;
        use antecedent_model::nonlinear_equilibrium::RegimeRow;
        let original = request();
        let mut traffic = original.components[1].clone();
        traffic.interface.produces[0].name = "traffic".into();
        traffic.interface.produces[0].quantity.variable_id = "game:traffic".into();
        traffic.interface.produces[0].quantity.variable_name = "traffic".into();
        traffic.interface.consumes[0].name = "from_wishlists".into();
        let mut wishlists = original.components[0].clone();
        wishlists.interface.produces[0].name = "wishlists".into();
        wishlists.interface.consumes[0].name = "from_traffic".into();
        wishlists.interface.consumes[0].quantity.variable_id = "game:traffic".into();
        wishlists.interface.consumes[0].quantity.variable_name = "traffic".into();
        let mut vector = VectorFeedbackRequest {
            components: vec![VectorFeedbackComponent {
                interface: InterfaceContract {
                    component: "game".into(),
                    produces: vec![
                        traffic.interface.produces[0].clone(),
                        wishlists.interface.produces[0].clone(),
                    ],
                    consumes: vec![
                        traffic.interface.consumes[0].clone(),
                        wishlists.interface.consumes[0].clone(),
                    ],
                },
                outputs: vec![
                    VectorFeedbackOutput {
                        produced_port: "traffic".into(),
                        equation: traffic.equation,
                        consumed_ports: vec!["from_wishlists".into()],
                    },
                    VectorFeedbackOutput {
                        produced_port: "wishlists".into(),
                        equation: wishlists.equation,
                        consumed_ports: vec!["from_traffic".into()],
                    },
                ],
            }],
            connections: vec![
                VectorFeedbackConnection {
                    ports: ComponentConnection::new("game", "traffic", "game", "from_traffic"),
                    consumer_output: "wishlists".into(),
                    coefficient: 0.3,
                },
                VectorFeedbackConnection {
                    ports: ComponentConnection::new("game", "wishlists", "game", "from_wishlists"),
                    consumer_output: "traffic".into(),
                    coefficient: 0.2,
                },
            ],
            noise_covariance: original.noise_covariance,
            noise: original.noise,
            initial: original.initial,
            worlds: original.worlds,
            policy: original.policy,
            factual: original.factual,
        };
        vector.noise_covariance = vec![0.04, 0.0, 0.0, 0.04];
        vector.worlds = vec![
            ComponentFeedbackWorld { clamps: vec![("game::traffic".into(), 0.0)] },
            ComponentFeedbackWorld { clamps: vec![("game::traffic".into(), 1.0)] },
        ];
        let training = (0..260)
            .map(|i| {
                let traffic = (i % 5) as f64 * 0.4;
                let disturbance = (i as f64 * 0.19).sin() * 0.2;
                RegimeRow {
                    values: vec![traffic, (1.0 + 0.3 * traffic + disturbance) / 0.8],
                    clamps: vec![Clamp { variable: 0, value: traffic }],
                }
            })
            .collect();
        let schedule = ComponentJointPosteriorSchedule {
            parameters: vec![CyclicParameter::Weight { equation: 1, parent: 0 }],
            training,
            prior_sd: 1.0,
            chains: 2,
            warmup: 300,
            draws_per_chain: 300,
            proposal_scale: 0.08,
        };
        let quantities: Vec<_> = ["ordinary", "promoted"]
            .iter()
            .map(|regime| {
                let mut q = quantity();
                q.regime_id = (*regime).into();
                q
            })
            .collect();
        let outputs: Vec<_> = quantities
            .iter()
            .enumerate()
            .map(|(world, quantity)| CyclicDecisionOutput {
                world,
                equation: 1,
                quantity: ScientificQuantityWire::from(quantity),
            })
            .collect();
        let contract = DecisionContract {
            actions: quantities
                .iter()
                .enumerate()
                .map(|(i, q)| DecisionAction {
                    id: ["ordinary", "promoted"][i].into(),
                    kind: ActionKind::Intervention,
                    inputs: vec![q.clone()],
                    utility: UtilityExpr::Input(0),
                })
                .collect(),
            utility_units: "count".into(),
            criterion: DecisionCriterion::ExpectedRegret,
            constraints: vec![],
            target_population: quantities[0].population_id.clone(),
            horizon: quantities[0].horizon,
            structural_policy: StructuralPolicy::RequireInvariantBestAction,
        };
        let ctx = ExecutionContext::for_tests(47);
        let plan = compile_joint_vector_component_posterior_decision(
            vector.clone(),
            schedule.clone(),
            outputs.clone(),
            contract.clone(),
            &ctx,
        )
        .unwrap();
        let bounded_quantities: Vec<_> = quantities
            .iter()
            .map(|q| {
                let mut bounded = q.clone();
                bounded.units = "dimensionless".into();
                bounded.transform_id = "logistic".into();
                bounded
            })
            .collect();
        let bounded_contract = DecisionContract {
            actions: bounded_quantities
                .iter()
                .enumerate()
                .map(|(i, q)| DecisionAction {
                    id: ["ordinary", "promoted"][i].into(),
                    kind: ActionKind::Intervention,
                    inputs: vec![q.clone()],
                    utility: UtilityExpr::Input(0),
                })
                .collect(),
            utility_units: "dimensionless".into(),
            criterion: DecisionCriterion::ExpectedRegret,
            constraints: vec![],
            target_population: bounded_quantities[0].population_id.clone(),
            horizon: bounded_quantities[0].horizon,
            structural_policy: StructuralPolicy::RequireInvariantBestAction,
        };
        let bounded = compile_bounded_component_posterior_decision(
            plan,
            bounded_quantities.clone(),
            bounded_contract.clone(),
        )
        .unwrap();
        assert_eq!(bounded.plan_view().route().as_str(), "BoundedComponentPosteriorDecision");
        let scope = bounded.reliability_scope().unwrap();
        let mut alternate_schedule = schedule;
        alternate_schedule.training[0].values[1] += 0.01;
        let alternate_ctx = ExecutionContext::for_tests(48);
        let alternate_inner = compile_joint_vector_component_posterior_decision(
            vector,
            alternate_schedule,
            outputs,
            contract,
            &alternate_ctx,
        )
        .unwrap();
        let alternate = compile_bounded_component_posterior_decision(
            alternate_inner,
            bounded_quantities,
            bounded_contract,
        )
        .unwrap();
        assert_eq!(scope, alternate.reliability_scope().unwrap());
        assert_ne!(bounded.lowered().identity(), alternate.lowered().identity());
        let mut result = bounded.execute(&ctx).unwrap();
        let (columns, _) = result.decision.chain.joint_output_law().unwrap();
        assert!(columns.iter().flatten().all(|bits| (0.0..=1.0).contains(&f64::from_bits(*bits))));
        assert_eq!(result.calibration, "not_measured");
        let artifact = bounded.export(&result, &ctx).unwrap();
        let replay = consume_bounded_component_posterior_decision(
            &artifact,
            bounded.lowered().identity(),
            &ctx,
        )
        .unwrap();
        assert_eq!(result.decision.decision, replay.decision.decision);
        let mut wire: BoundedComponentPosteriorWire = antecedent_io::from_cbor(&artifact).unwrap();
        wire.quantities[0].transform_id = "identity".into();
        let changed = antecedent_io::to_cbor(&wire).unwrap();
        assert!(
            consume_bounded_component_posterior_decision(
                &changed,
                bounded.lowered().identity(),
                &ctx
            )
            .is_err()
        );
        result.inner.child.posterior.samples[0][0] += 1.0;
        assert!(bounded.export(&result, &ctx).is_err());
    }
}
