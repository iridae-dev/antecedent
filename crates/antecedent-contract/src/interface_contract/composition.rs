//! A set of components and the connections between them: checked as a whole
//! before any execution.
//!
//! Beyond the pairwise decision of [`super::check_connection`], a composition
//! has properties no pair has. A connection that closes a loop (feedback
//! between one component's output and another's input) is refused with the
//! cycle path named: cyclic scientific composition is not supported, and a
//! feedback loop is never silently unrolled. A calibration declared inherited
//! is resolved against the upstream producer and refused when the upstream
//! does not hold it. Inputs that must rest on disjoint evidence are checked
//! pairwise against the sources their producers declare, so one observation
//! cannot reach a consumer twice. Without a ledger that comparison is of the
//! declared source identities; [`check_composition_with_ledger`] decides it
//! from the evidence ledger's row-level overlap, so two distinct sources that
//! share observations (a subset, a resample, a derived source) are refused
//! and a pair whose overlap is not computable is unknown, never disjoint.
//!
//! The work is linear in the declared sizes (up to ordering), so there is no search to budget
//! or cancel and no count bound to enforce; [`MAX_COMPOSITION_COMPONENTS`] and
//! [`MAX_COMPOSITION_CONNECTIONS`] are only the defaults a decoder of untrusted input applies.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{BTreeMap, BTreeSet};

use antecedent_core::identity::{CanonicalWriter, SemanticDigest};

use super::connection::{Connection, check_ports};
use super::declaration::{
    CalibrationStatus, DependenceRequirement, EvidenceBasis, INTERFACE_CONTRACT_VERSION,
    InterfaceContract,
};
use super::refusal::{
    InterfaceRefusal, calibration_inherited_not_held, calibration_inherited_quantity_changed,
    calibration_inherited_unfed, calibration_without_interval, cyclic_composition,
    dependence_unknown, duplicate_component, evidence_overlap_unknown, port_multiply_fed,
    shared_evidence, unknown_component, unknown_port,
};
use crate::evidence_ledger::{EvidenceLedger, Overlap, SourceId};

/// Default bound on components in a composition record that a decoder of untrusted input applies.
pub const MAX_COMPOSITION_COMPONENTS: usize = 1024;
/// Default bound on connections in a composition record that a decoder of untrusted input
/// applies.
pub const MAX_COMPOSITION_CONNECTIONS: usize = 8192;

const TAG_COMPOSITION: &str = "antecedent.interface_contract.composition";

/// One declared connection: a produced port of one component feeds a
/// consumed port of another.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComponentConnection {
    /// Producing component.
    pub producer: String,
    /// Produced port of the producer.
    pub produced_port: String,
    /// Consuming component.
    pub consumer: String,
    /// Consumed port of the consumer.
    pub consumed_port: String,
}

impl ComponentConnection {
    /// A connection from `producer.produced_port` to `consumer.consumed_port`.
    #[must_use]
    pub fn new(
        producer: impl Into<String>,
        produced_port: impl Into<String>,
        consumer: impl Into<String>,
        consumed_port: impl Into<String>,
    ) -> Self {
        Self {
            producer: producer.into(),
            produced_port: produced_port.into(),
            consumer: consumer.into(),
            consumed_port: consumed_port.into(),
        }
    }
}

/// A composition every connection of which is licensed by the declarations
/// and which has no cycle.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComposedInterfaces {
    /// Components in a feed-forward order (producers before consumers; ties
    /// broken by component name).
    pub order: Vec<String>,
    /// The accepted connections, ordered by consumer, then consumed port.
    pub connections: Vec<Connection>,
    /// Identity binding every component identity, connection and the order.
    pub identity: SemanticDigest,
}

type Edges<'a> = BTreeMap<&'a str, BTreeSet<&'a str>>;

/// A feed-forward order of the components, or the cycle path that prevents one.
fn feed_forward_order<'a>(
    names: &BTreeSet<&'a str>,
    successors: &Edges<'a>,
    predecessors: &Edges<'a>,
) -> Result<Vec<&'a str>, String> {
    let mut indegree: BTreeMap<&str, usize> =
        names.iter().map(|name| (*name, predecessors.get(name).map_or(0, BTreeSet::len))).collect();
    let mut ready: BTreeSet<&str> =
        indegree.iter().filter(|(_, degree)| **degree == 0).map(|(name, _)| *name).collect();
    let mut order = Vec::with_capacity(names.len());
    while let Some(next) = ready.pop_first() {
        order.push(next);
        for successor in successors.get(next).into_iter().flatten() {
            if let Some(degree) = indegree.get_mut(successor) {
                *degree -= 1;
                if *degree == 0 {
                    ready.insert(*successor);
                }
            }
        }
    }
    if order.len() == names.len() {
        return Ok(order);
    }
    // Every component left over has a predecessor among the leftovers; walking
    // predecessors from the smallest name must repeat a component, which closes
    // a cycle.
    let placed: BTreeSet<&str> = order.iter().copied().collect();
    let remaining: BTreeSet<&str> = names.difference(&placed).copied().collect();
    let mut walk: Vec<&str> = Vec::new();
    let mut walked: BTreeSet<&str> = BTreeSet::new();
    let mut current = remaining.iter().next().copied().unwrap_or_default();
    while !walked.contains(&current) {
        walk.push(current);
        walked.insert(current);
        let previous = predecessors
            .get(current)
            .and_then(|set| set.iter().copied().find(|name| remaining.contains(name)));
        match previous {
            Some(name) => current = name,
            None => break,
        }
    }
    let start = walk.iter().position(|name| *name == current).unwrap_or(0);
    let mut cycle: Vec<&str> = walk[start..].to_vec();
    cycle.reverse();
    // Start the path at the smallest name so one cycle has one spelling.
    if let Some(smallest) = (0..cycle.len()).min_by_key(|index| cycle[*index]) {
        cycle.rotate_left(smallest);
    }
    let first = cycle.first().copied().unwrap_or_default();
    cycle.push(first);
    Err(cycle.join(" -> "))
}

/// Check a set of components and the connections between them.
///
/// The checks run in a fixed order and stop at the first group that makes
/// the next meaningless: bounds and malformed declarations, then unknown
/// components or ports and consumed ports fed twice, then a cycle (reported
/// alone, with its path), then the pairwise decision of every connection with
/// inherited calibration resolved in feed-forward order, then disjoint-evidence
/// requirements.
///
/// # Errors
/// Every defect of the group that failed, each a coded refusal with an
/// `interface_contract.*` detail and the declaration that would license it.
pub fn check_composition(
    components: &[InterfaceContract],
    connections: &[ComponentConnection],
) -> Result<ComposedInterfaces, Vec<InterfaceRefusal>> {
    check_composition_inner(components, connections, None)
}

/// As [`check_composition`], deciding the disjoint-evidence requirements from
/// the evidence ledger's row-level overlap of the declared sources.
///
/// For every pair of inputs that require disjoint evidence, every declared
/// source of one is compared with every declared source of the other: the same
/// source, or two sources the ledger finds to share observations, refuse as
/// `interface_contract.shared_evidence`; a source the ledger does not hold, or
/// whose observations are not computable, refuses as
/// `interface_contract.dependence_unknown`. Unknown overlap is never disjoint.
///
/// # Errors
/// As [`check_composition`].
pub fn check_composition_with_ledger(
    components: &[InterfaceContract],
    connections: &[ComponentConnection],
    ledger: &EvidenceLedger,
) -> Result<ComposedInterfaces, Vec<InterfaceRefusal>> {
    check_composition_inner(components, connections, Some(ledger))
}

fn check_composition_inner(
    components: &[InterfaceContract],
    connections: &[ComponentConnection],
    ledger: Option<&EvidenceLedger>,
) -> Result<ComposedInterfaces, Vec<InterfaceRefusal>> {
    let by_name = validate_components(components)?;
    resolve_connections(connections, &by_name)?;
    let order = order_components(&by_name, connections)?;
    let mut refusals = Vec::new();
    let effective = resolve_inherited_calibration(&order, &by_name, connections, &mut refusals);
    let accepted = decide_connections(connections, &by_name, &effective, &mut refusals);
    // Inputs required to rest on disjoint evidence must not share a source.
    check_disjoint(&accepted, &by_name, ledger, &mut refusals);
    if !refusals.is_empty() {
        return Err(refusals);
    }
    Ok(compose(&by_name, &order, accepted))
}

/// Phase 1: every component is well formed and named once.
fn validate_components(
    components: &[InterfaceContract],
) -> Result<BTreeMap<&str, &InterfaceContract>, Vec<InterfaceRefusal>> {
    let mut refusals = Vec::new();
    let mut by_name: BTreeMap<&str, &InterfaceContract> = BTreeMap::new();
    for contract in components {
        if let Err(refusal) = contract.validate() {
            refusals.push(refusal);
        } else if by_name.insert(contract.component.as_str(), contract).is_some() {
            refusals.push(duplicate_component(&contract.component));
        }
    }
    if refusals.is_empty() { Ok(by_name) } else { Err(refusals) }
}

/// Phase 2: every connection names declared components and ports, and no consumed port is fed
/// twice.
fn resolve_connections(
    connections: &[ComponentConnection],
    by_name: &BTreeMap<&str, &InterfaceContract>,
) -> Result<(), Vec<InterfaceRefusal>> {
    let mut refusals = Vec::new();
    let mut fed: BTreeMap<(&str, &str), usize> = BTreeMap::new();
    for connection in connections {
        let producer = by_name.get(connection.producer.as_str());
        let consumer = by_name.get(connection.consumer.as_str());
        match producer {
            None => refusals.push(unknown_component(&connection.producer)),
            Some(contract) => {
                if contract.produced(&connection.produced_port).is_none() {
                    refusals.push(unknown_port(&connection.producer, &connection.produced_port));
                }
            }
        }
        match consumer {
            None => refusals.push(unknown_component(&connection.consumer)),
            Some(contract) => {
                if contract.consumed(&connection.consumed_port).is_none() {
                    refusals.push(unknown_port(&connection.consumer, &connection.consumed_port));
                } else {
                    *fed.entry((
                        connection.consumer.as_str(),
                        connection.consumed_port.as_str(),
                    ))
                    .or_insert(0) += 1;
                }
            }
        }
    }
    for ((component, port), count) in &fed {
        if *count > 1 {
            refusals.push(port_multiply_fed(component, port));
        }
    }
    if refusals.is_empty() { Ok(()) } else { Err(refusals) }
}

/// Phase 3: a feed-forward order, or the cycle that prevents one. Cycles are refused before any
/// pairwise decision or execution.
fn order_components<'a>(
    by_name: &BTreeMap<&'a str, &InterfaceContract>,
    connections: &'a [ComponentConnection],
) -> Result<Vec<&'a str>, Vec<InterfaceRefusal>> {
    let names: BTreeSet<&str> = by_name.keys().copied().collect();
    let mut successors: Edges<'_> = BTreeMap::new();
    let mut predecessors: Edges<'_> = BTreeMap::new();
    for connection in connections {
        successors
            .entry(connection.producer.as_str())
            .or_default()
            .insert(connection.consumer.as_str());
        predecessors
            .entry(connection.consumer.as_str())
            .or_default()
            .insert(connection.producer.as_str());
    }
    feed_forward_order(&names, &successors, &predecessors)
        .map_err(|path| vec![cyclic_composition(path)])
}

/// Phase 4: the calibration each produced port effectively holds, resolving `Inherited`
/// against the upstream producer in feed-forward order.
fn resolve_inherited_calibration<'a>(
    order: &[&'a str],
    by_name: &BTreeMap<&'a str, &'a InterfaceContract>,
    connections: &'a [ComponentConnection],
    refusals: &mut Vec<InterfaceRefusal>,
) -> BTreeMap<(&'a str, &'a str), CalibrationStatus> {
    let feeding: BTreeMap<(&str, &str), &ComponentConnection> =
        connections.iter().map(|c| ((c.consumer.as_str(), c.consumed_port.as_str()), c)).collect();
    let mut effective: BTreeMap<(&str, &str), CalibrationStatus> = BTreeMap::new();
    for component in order {
        let Some(contract) = by_name.get(component) else { continue };
        for port in &contract.produces {
            let status = match &port.calibration {
                CalibrationStatus::Inherited { input } => {
                    match feeding.get(&(*component, input.as_str())) {
                        None => {
                            refusals.push(calibration_inherited_unfed(&port.name, input));
                            CalibrationStatus::Unmeasured
                        }
                        Some(upstream) => {
                            // A retained unresolved declaration is inspectable, but it may
                            // resolve to Measured only for a calibratable operation.
                            if !port
                                .operations
                                .contains(&super::declaration::ContractOperation::Interval)
                                && !port
                                    .operations
                                    .contains(&super::declaration::ContractOperation::Posterior)
                            {
                                refusals.push(calibration_without_interval(&port.name));
                                effective.insert(
                                    (*component, port.name.as_str()),
                                    CalibrationStatus::Unmeasured,
                                );
                                continue;
                            }
                            let source = by_name
                                .get(upstream.producer.as_str())
                                .and_then(|component| component.produced(&upstream.produced_port));
                            // An input port states requirements, which may be less specific
                            // than the supplied quantity. Inheritance follows the actual
                            // upstream output, including its uncertainty and evidence.
                            if source.is_none_or(|source| {
                                port.quantity.require_same_coordinate(&source.quantity).is_err()
                                    || port.semantic_identity != source.semantic_identity
                                    || port.support != source.support
                                    || port.uncertainty != source.uncertainty
                                    || !port.operations.is_subset(&source.operations)
                                    || port.measurement_models != source.measurement_models
                                    || port.evidence != source.evidence
                            }) {
                                refusals.push(calibration_inherited_quantity_changed(
                                    &port.name, input,
                                ));
                                effective.insert(
                                    (*component, port.name.as_str()),
                                    CalibrationStatus::Unmeasured,
                                );
                                continue;
                            }
                            let held = effective
                                .get(&(upstream.producer.as_str(), upstream.produced_port.as_str()))
                                .cloned()
                                .unwrap_or(CalibrationStatus::Unmeasured);
                            if !held.is_measured() {
                                refusals.push(calibration_inherited_not_held(
                                    &port.name,
                                    input,
                                    held.label().to_owned(),
                                ));
                            }
                            held
                        }
                    }
                }
                other => other.clone(),
            };
            effective.insert((*component, port.name.as_str()), status);
        }
    }
    effective
}

/// Phase 5: the pairwise decision of every connection, in canonical (consumer, consumed port)
/// order.
fn decide_connections(
    connections: &[ComponentConnection],
    by_name: &BTreeMap<&str, &InterfaceContract>,
    effective: &BTreeMap<(&str, &str), CalibrationStatus>,
    refusals: &mut Vec<InterfaceRefusal>,
) -> Vec<Connection> {
    let mut ordered: Vec<&ComponentConnection> = connections.iter().collect();
    ordered.sort_by(|a, b| {
        (a.consumer.as_str(), a.consumed_port.as_str())
            .cmp(&(b.consumer.as_str(), b.consumed_port.as_str()))
    });
    let mut accepted = Vec::with_capacity(ordered.len());
    for connection in &ordered {
        let (Some(producer), Some(consumer)) =
            (by_name.get(connection.producer.as_str()), by_name.get(connection.consumer.as_str()))
        else {
            continue;
        };
        let (Some(out), Some(inp)) = (
            producer.produced(&connection.produced_port),
            consumer.consumed(&connection.consumed_port),
        ) else {
            continue;
        };
        let resolved =
            effective.get(&(connection.producer.as_str(), connection.produced_port.as_str()));
        match check_ports(producer, out, consumer, inp, resolved) {
            Ok(accepted_connection) => accepted.push(accepted_connection),
            Err(found) => refusals.extend(found),
        }
    }
    accepted
}

/// Phase 6: the composed result and the identity binding every component, connection and the
/// order.
fn compose(
    by_name: &BTreeMap<&str, &InterfaceContract>,
    order: &[&str],
    accepted: Vec<Connection>,
) -> ComposedInterfaces {
    let mut writer = CanonicalWriter::new(TAG_COMPOSITION);
    writer.word(INTERFACE_CONTRACT_VERSION).word(by_name.len() as u64);
    for (name, contract) in by_name {
        writer.text(name).text(&contract.canonical_identity().to_hex());
    }
    writer.word(accepted.len() as u64);
    for connection in &accepted {
        writer.text(&connection.identity.to_hex());
    }
    writer.word(order.len() as u64);
    for name in order {
        writer.text(name);
    }
    ComposedInterfaces {
        order: order.iter().map(|name| (*name).to_owned()).collect(),
        connections: accepted,
        identity: SemanticDigest::from_bytes(writer.finish()),
    }
}

const TAG_CYCLIC_COMPOSITION: &str = "antecedent.interface_contract.composition.cyclic_components";

/// A certified cyclic evidence product admitted as one component of a composition.
///
/// The product's internal feedback is solved inside the component under its
/// declared solve contract; it is not a cycle among components. Connections to
/// and from the component are checked like any other, and a cycle among
/// components is still refused with its path.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CyclicComponentDeclaration {
    /// Component the product is declared as. It names exactly one declared contract.
    pub component: String,
    /// Program identity of the cyclic product: its contract's program layer.
    pub program: SemanticDigest,
    /// `cyclic.solution.v1` tag its answers were solved under.
    pub cyclic_solution: SemanticDigest,
}

/// As [`check_composition`] (or [`check_composition_with_ledger`] when `ledger` is
/// given), admitting each certified cyclic product in `cyclic` as one declared
/// component.
///
/// Each declaration names one declared component, and a component carries at
/// most one cyclic product. The composition identity also binds every declared
/// product's program identity and cyclic-solution tag, so the same interfaces
/// over another cyclic product are a different composition.
///
/// # Errors
/// A declaration that names no declared component or repeats one, then every
/// defect [`check_composition`] reports. A cycle among components is refused.
pub fn check_composition_with_cyclic_components(
    components: &[InterfaceContract],
    connections: &[ComponentConnection],
    cyclic: &[CyclicComponentDeclaration],
    ledger: Option<&EvidenceLedger>,
) -> Result<ComposedInterfaces, Vec<InterfaceRefusal>> {
    let declared: BTreeSet<&str> =
        components.iter().map(|contract| contract.component.as_str()).collect();
    let mut refusals = Vec::new();
    let mut seen = BTreeSet::new();
    for product in cyclic {
        if !declared.contains(product.component.as_str()) {
            refusals.push(unknown_component(&product.component));
        } else if !seen.insert(product.component.as_str()) {
            refusals.push(duplicate_component(&product.component));
        }
    }
    if !refusals.is_empty() {
        return Err(refusals);
    }
    let composed = check_composition_inner(components, connections, ledger)?;
    if cyclic.is_empty() {
        return Ok(composed);
    }
    let mut ordered: Vec<&CyclicComponentDeclaration> = cyclic.iter().collect();
    ordered.sort_by(|left, right| left.component.cmp(&right.component));
    let mut writer = CanonicalWriter::new(TAG_CYCLIC_COMPOSITION);
    writer.text(&composed.identity.to_hex()).word(ordered.len() as u64);
    for product in ordered {
        writer
            .text(&product.component)
            .text(&product.program.to_hex())
            .text(&product.cyclic_solution.to_hex());
    }
    Ok(ComposedInterfaces { identity: SemanticDigest::from_bytes(writer.finish()), ..composed })
}

/// The consumed ports that require disjoint evidence among `components`, as
/// refusals for a caller that has no evidence ledger.
///
/// Without a ledger the overlap of two declared sources is not computable, so
/// a disjointness requirement cannot be decided from identities alone; a
/// caller without a ledger refuses with these rather than skip the
/// requirement. Empty when no consumed port requires disjoint evidence.
#[must_use]
pub fn disjoint_requirements_without_ledger(
    components: &[InterfaceContract],
) -> Vec<InterfaceRefusal> {
    components
        .iter()
        .flat_map(|component| {
            component
                .consumes
                .iter()
                .filter(|port| port.dependence == DependenceRequirement::DisjointEvidence)
                .map(|port| dependence_unknown(&format!("{}.{}", component.component, port.name)))
        })
        .collect()
}

fn check_disjoint(
    accepted: &[Connection],
    by_name: &BTreeMap<&str, &InterfaceContract>,
    ledger: Option<&EvidenceLedger>,
    refusals: &mut Vec<InterfaceRefusal>,
) {
    let mut per_consumer: BTreeMap<&str, Vec<&Connection>> = BTreeMap::new();
    for connection in accepted {
        let requires_disjoint = by_name
            .get(connection.consumer.as_str())
            .and_then(|contract| contract.consumed(&connection.consumed_port))
            .is_some_and(|port| port.dependence == DependenceRequirement::DisjointEvidence);
        if requires_disjoint {
            per_consumer.entry(connection.consumer.as_str()).or_default().push(connection);
        }
    }
    for inputs in per_consumer.values() {
        for (index, first) in inputs.iter().enumerate() {
            for second in &inputs[index + 1..] {
                let (EvidenceBasis::Declared(a), EvidenceBasis::Declared(b)) =
                    (&first.evidence, &second.evidence)
                else {
                    continue;
                };
                let (first_name, second_name) = (
                    format!("{}.{}", first.producer, first.produced_port),
                    format!("{}.{}", second.producer, second.produced_port),
                );
                let (shared, undecided) = match ledger {
                    // Equal sources establish overlap. Distinct identities alone
                    // cannot establish that the underlying observations are disjoint.
                    None => (a.intersection(b).count(), true),
                    Some(ledger) => ledger_overlap(a, b, ledger),
                };
                if shared > 0 {
                    refusals.push(shared_evidence(&first_name, &second_name, shared));
                } else if undecided {
                    refusals.push(evidence_overlap_unknown(&first_name, &second_name));
                }
            }
        }
    }
}

/// Observations the two declared source sets share per the ledger (identical
/// sources count one observation set each), and whether any pair could not be
/// decided.
fn ledger_overlap(
    a: &BTreeSet<SourceId>,
    b: &BTreeSet<SourceId>,
    ledger: &EvidenceLedger,
) -> (usize, bool) {
    let mut shared: usize = 0;
    let mut undecided = false;
    for left in a {
        for right in b {
            if left == right {
                shared += 1;
                continue;
            }
            match ledger.overlap(*left, *right) {
                Ok(Overlap::Independent) => {}
                Ok(Overlap::SharedObservations { shared: rows }) => {
                    shared += usize::try_from(rows).unwrap_or(usize::MAX).max(1);
                }
                Ok(Overlap::Unknown { .. }) | Err(_) => undecided = true,
            }
        }
    }
    (shared, undecided)
}
