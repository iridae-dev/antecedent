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

use antecedent_contract::interface_contract::{
    CalibrationStatus, ComponentConnection, DependenceRequirement, EvidenceBasis, InterfaceContract,
    InterfaceRefusal, check_connection,
};
use antecedent_contract::evidence_ledger::{EvidenceLedger, Overlap};
use antecedent_core::{CanonicalWriter, ExecutionContext, SemanticDigest};
use antecedent_io::cyclic_system_artifact::{CyclicArtifactError, CyclicExecutionRequest};
use antecedent_model::nonlinear_equilibrium::{
    Activation, Clamp, Equation, EquilibriumModel, SolvePolicy,
};

use super::cyclic::CyclicPreparedState;
use super::{PreparedStudy, Study};

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
        component.interface.validate().map_err(|refusal| ComponentFeedbackError::Interface(vec![refusal]))?;
        if component.outputs.len() != component.interface.produces.len() {
            return Err(ComponentFeedbackError::Invalid("each produced port needs exactly one output equation"));
        }
        let name = component.interface.component.clone();
        let mut seen_outputs = BTreeSet::new();
        for output in component.outputs {
            if !seen_outputs.insert(output.produced_port.clone()) {
                return Err(ComponentFeedbackError::Invalid("an output equation is repeated"));
            }
            let port = component.interface.produces.iter()
                .find(|port| port.name == output.produced_port)
                .ok_or(ComponentFeedbackError::Invalid("an output equation names no produced port"))?;
            let full = format!("{name}::{}", port.name);
            if produced.insert((name.clone(), port.name.clone()), full.clone()).is_some() {
                return Err(ComponentFeedbackError::Invalid("a produced port is repeated"));
            }
            let mut inputs = Vec::new();
            for input_name in &output.consumed_ports {
                let input = component.interface.consumes.iter()
                    .find(|input| input.name == *input_name)
                    .ok_or(ComponentFeedbackError::Invalid("an output reads an undeclared consumed port"))?;
                if consumed.insert((name.clone(), port.name.clone(), input.name.clone()), full.clone()).is_some() {
                    return Err(ComponentFeedbackError::Invalid("a consumed port is repeated within one output equation"));
                }
                inputs.push(input.clone());
            }
            scalar.push(ScalarFeedbackComponent {
                interface: InterfaceContract {
                    component: full,
                    produces: vec![port.clone()], consumes: inputs,
                },
                equation: output.equation,
            });
        }
        if component.interface.consumes.iter().any(|port| {
            !consumed.keys().any(|(component, _, input)| component == &name && input == &port.name)
        }) {
            return Err(ComponentFeedbackError::Invalid("every consumed port needs one output equation"));
        }
    }
    let mut connections = Vec::with_capacity(request.connections.len());
    for connection in request.connections {
        let source = produced.get(&(
            connection.ports.producer.clone(), connection.ports.produced_port.clone(),
        )).ok_or(ComponentFeedbackError::Invalid("a connection names no produced output"))?;
        let target = consumed.get(&(
            connection.ports.consumer.clone(), connection.consumer_output.clone(),
            connection.ports.consumed_port.clone(),
        )).ok_or(ComponentFeedbackError::Invalid("a connection names no consumed output"))?;
        connections.push(FeedbackConnection {
            ports: ComponentConnection::new(
                source.clone(), connection.ports.produced_port.clone(),
                target.clone(), connection.ports.consumed_port.clone(),
            ),
            coefficient: connection.coefficient,
        });
    }
    prepare_component_feedback_inner(ComponentFeedbackRequest {
        components: scalar, connections,
        noise_covariance: request.noise_covariance, noise: request.noise,
        initial: request.initial, worlds: request.worlds,
        policy: request.policy, factual: request.factual,
    }, ledger, ctx)
}

fn check_aggregate_evidence(
    components: &[&InterfaceContract],
    connections: &[&ComponentConnection],
    ledger: Option<&EvidenceLedger>,
) -> Result<(), ComponentFeedbackError> {
    let by_name: BTreeMap<&str, &InterfaceContract> = components.iter()
        .map(|contract| (contract.component.as_str(), *contract)).collect();
    for consumer in components {
        let inputs: Vec<_> = consumer.consumes.iter()
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
                    let producer = by_name.get(connection.producer.as_str())
                        .ok_or(ComponentFeedbackError::Invalid("an evidence producer is unknown"))?;
                    let output = producer.produces.iter()
                        .find(|port| port.name == connection.produced_port)
                        .ok_or(ComponentFeedbackError::Invalid("an evidence output is unknown"))?;
                    match &output.evidence {
                        EvidenceBasis::Declared(sources) if !sources.is_empty() => Ok(sources.clone()),
                        _ => Err(ComponentFeedbackError::Invalid(
                            "disjoint-evidence inputs need declared producer sources",
                        )),
                    }
                };
                let (left, right) = (sources(&first.name)?, sources(&second.name)?);
                let ledger = ledger.ok_or(ComponentFeedbackError::Invalid(
                    "multiple disjoint-evidence inputs require an evidence ledger",
                ))?;
                for a in left {
                    for b in right {
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
    request.components.sort_by(|left, right| {
        left.interface.component.cmp(&right.interface.component)
    });
    let order: Vec<String> = request.components.iter()
        .map(|component| component.interface.component.clone()).collect();
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
    let positions: BTreeMap<&str, usize> = order.iter().enumerate()
        .map(|(index, name)| (name.as_str(), index)).collect();
    let mut writer = CanonicalWriter::new(COMPOSITION_TAG);
    writer.word(order.len() as u64);
    if let Some(ledger) = ledger {
        writer.bytes(ledger.state_digest().as_bytes());
    }
    for component in &request.components {
        component.interface.validate().map_err(|refusal| {
            ComponentFeedbackError::Interface(vec![refusal])
        })?;
        if component.interface.produces.len() != 1
            || component.equation.weights.len() != 1
            || component.equation.expression.is_some()
            || !component.equation.parameters.is_empty()
        {
            return Err(ComponentFeedbackError::Invalid(
                "each component needs one produced port and one scalar local equation",
            ));
        }
        component.equation.validate(1).map_err(|_| {
            ComponentFeedbackError::Invalid("a local scalar equation is invalid")
        })?;
        if component.interface.produces.iter().any(|port| {
            matches!(&port.calibration, CalibrationStatus::Measured { .. } | CalibrationStatus::Inherited { .. })
        }) {
            return Err(ComponentFeedbackError::Invalid(
                "a measured or inherited component calibration does not cover the connected equilibrium",
            ));
        }
        if component.interface.consumes.iter().filter(|port| {
            port.dependence == DependenceRequirement::DisjointEvidence
        }).count() > 1 {
            return Err(ComponentFeedbackError::Invalid(
                "multiple disjoint-evidence inputs require a joint evidence-ledger check",
            ));
        }
        writer.text(&component.interface.component)
            .text(&component.interface.canonical_identity().to_hex())
            .text(&component.equation.id)
            .word(component.equation.intercept.to_bits())
            .word(component.equation.weights[0].to_bits())
            .word(activation_tag(component.equation.activation));
    }
    request.connections.sort_by(|left, right| {
        (&left.ports.consumer, &left.ports.consumed_port,
         &left.ports.producer, &left.ports.produced_port)
            .cmp(&(&right.ports.consumer, &right.ports.consumed_port,
                   &right.ports.producer, &right.ports.produced_port))
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
        if producer == consumer || !connection.coefficient.is_finite()
            || connection.coefficient == 0.0
            || !fed.insert((consumer, ports.consumed_port.as_str()))
            || !matrix_edges.insert((consumer, producer))
        {
            return Err(ComponentFeedbackError::Invalid(
                "a cross-component input must be distinct, nonzero and occupy one structural coefficient",
            ));
        }
        let checked = check_connection(
            &request.components[producer].interface, &ports.produced_port,
            &request.components[consumer].interface, &ports.consumed_port,
        ).map_err(ComponentFeedbackError::Interface)?;
        writer.text(&checked.identity.to_hex()).word(connection.coefficient.to_bits());
        couplings.push((consumer, producer, connection.coefficient));
    }
    for (index, component) in request.components.iter().enumerate() {
        if component.interface.consumes.iter().any(|port| {
            !fed.contains(&(index, port.name.as_str()))
        }) {
            return Err(ComponentFeedbackError::Invalid(
                "every declared consumed port needs one structural connection",
            ));
        }
    }
    let composition_identity = SemanticDigest::from_bytes(writer.finish());
    let n = order.len();
    let mut equations: Vec<Equation> = request.components.into_iter().enumerate().map(|(index, component)| {
        let mut equation = component.equation;
        let self_weight = equation.weights[0];
        equation.id = format!("component.{index}.{}", composition_identity.to_hex());
        equation.weights = vec![0.0; n];
        equation.weights[index] = self_weight;
        equation
    }).collect();
    for (consumer, producer, coefficient) in couplings {
        equations[consumer].weights[producer] = coefficient;
    }
    let worlds = request.worlds.into_iter().map(|world| {
        let mut seen = BTreeSet::new();
        world.clamps.into_iter().map(|(name, value)| {
            let variable = *positions.get(name.as_str()).ok_or(
                ComponentFeedbackError::Invalid("a clamp names an unknown component")
            )?;
            if !value.is_finite() || !seen.insert(variable) {
                return Err(ComponentFeedbackError::Invalid(
                    "a world clamps each component at most once to a finite value",
                ));
            }
            Ok(Clamp { variable, value })
        }).collect::<Result<Vec<_>, _>>()
    }).collect::<Result<Vec<_>, _>>()?;
    let prepared = Study::prepare_cyclic(CyclicExecutionRequest {
        model: EquilibriumModel { equations, noise_covariance: request.noise_covariance },
        fit: None,
        worlds,
        noise: request.noise,
        initial: request.initial,
        policy: request.policy,
        factual: request.factual,
    }, ctx)?;
    Ok(PreparedComponentFeedback { prepared, order, composition_identity })
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_contract::evidence_ledger::{ObservationId, SourceDeclaration, SourceKind};
    use antecedent_contract::{QuantityRole, ScientificQuantity};
    use antecedent_contract::interface_contract::{
        ConsumedPort, ContractOperation, ProducedPort, SupportDomain,
    };
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
                    "out", quantity(), "model:specified", SupportDomain::Real,
                    [ContractOperation::Mean],
                )],
                consumes: vec![ConsumedPort::new(
                    "in", quantity(), "model:specified", SupportDomain::Real,
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
        let first = ledger.add_source(SourceDeclaration::new("ads", SourceKind::ObservationSet)
            .with_observations([ObservationId::from_key("campaign", b"ads")]))
            .expect("first source");
        let second = ledger.add_source(SourceDeclaration::new("community", SourceKind::ObservationSet)
            .with_observations([ObservationId::from_key("campaign", b"community")]))
            .expect("second source");
        let mut draft = request();
        draft.components.push(component("community", 0.1));
        draft.components[0].interface.consumes.push(
            ConsumedPort::new("community_in", quantity(), "model:specified",
                SupportDomain::Real, [ContractOperation::Mean])
                .with_dependence(DependenceRequirement::DisjointEvidence),
        );
        draft.components[0].interface.consumes[0].dependence = DependenceRequirement::DisjointEvidence;
        draft.components[1].interface.produces[0].evidence = EvidenceBasis::Declared([first].into());
        draft.components[2].interface.produces[0].evidence = EvidenceBasis::Declared([second].into());
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
        changed_ledger.add_source(SourceDeclaration::new("later", SourceKind::ObservationSet)
            .with_observations([ObservationId::from_key("campaign", b"later")]))
            .expect("later source");
        let changed = prepare_component_feedback_with_ledger(draft.clone(), &changed_ledger, &ctx)
            .expect("changed ledger still proves disjointness");
        assert_ne!(prepared.prepared().lowered().identity(), changed.prepared().lowered().identity());
        draft.components[2].interface.produces[0].evidence = EvidenceBasis::Declared([first].into());
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
                    produces: vec![traffic.interface.produces[0].clone(),
                                   wishlists.interface.produces[0].clone()],
                    consumes: vec![traffic.interface.consumes[0].clone(),
                                   wishlists.interface.consumes[0].clone()],
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
            noise: original.noise, initial: original.initial,
            worlds: original.worlds, policy: original.policy, factual: original.factual,
        };
        let ctx = ExecutionContext::for_tests(37);
        let prepared = prepare_vector_component_feedback(vector, &ctx)
            .expect("multi-output cyclic compiler");
        assert_eq!(prepared.component_order(), &[
            "game::traffic".to_owned(), "game::wishlists".to_owned(),
        ]);
        let equations = &prepared.prepared().request().model.equations;
        assert_eq!(equations[0].weights, [0.1, 0.2]);
        assert_eq!(equations[1].weights, [0.3, 0.2]);
    }
}
