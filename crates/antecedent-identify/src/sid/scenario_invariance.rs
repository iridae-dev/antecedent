//! Per-scenario selection differences and invariances (2.3.0 A1 remainder).
//!
//! A transport scenario is a selection diagram: a graph plus the selection
//! targets, the variables whose generating mechanism may differ between the
//! source and the target. Every answer a scenario gives rests on two kinds of
//! assumption, and this module reports both:
//!
//! * the **selection differences**: the selection targets, the shared
//!   mechanisms (every other variable) and the directed and bidirected edges the
//!   scenario assumes;
//! * the **invariances the identified formula relies on**: each factor the
//!   formula takes from the SOURCE population, with its variables, the
//!   conditioning variables, the experimental regime (the `do` set and the
//!   catalog regime id) and the rule of the proof step that produced it. The
//!   mechanisms of the factor's variables are assumed shared between source and
//!   target; the checker's S-admissibility test (run twice, by two independent
//!   implementations) has already established that no selection node reaches
//!   them in the mutilated graph, which is exactly the statement that they are
//!   not selection targets and are not selection-connected.
//!
//! # Extraction
//!
//! Nothing here re-decides admissibility or re-derives a formula. The factors
//! come from the checked [`ClassicalTransportDerivation`] itself: the formula's
//! distribution leaves under its root expression (the expression the checker
//! verified equal to the root proof step's output) are partitioned by
//! population, and each source leaf is attributed to the reachable leaf-producing
//! proof step (`sid.line10`, `transport.direct` or
//! `transport.pretreatment_standardize`) whose output contains it. A source leaf
//! that no such step produced is a contradiction between the formula and the
//! proof and is refused. The catalog regime id is read from the catalog-bound
//! copy of the same leaf. The only graph computation is a descriptive one: the
//! selection targets inside the leaf's bidirected district, which the checked
//! admissibility already accounts for (they can only occur where conditioning
//! handles them).
//!
//! A scenario that is not identified reports its structural reason instead (the
//! s-hedge or witness variables), never an invariance list.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeSet;
use std::fmt::Write as _;
use std::sync::Arc;

use antecedent_core::{NodeRef, VariableId, reason_code};
use antecedent_expr::LeafSignature;
use antecedent_graph::{DenseNodeId, SelectionDiagram};

use super::scenarios::{ScenarioOutcome, TransportScenario};
use super::{BoundTransportFunctional, ClassicalTransportDerivation, Rule, SHedgeRecord};

/// Detail of a report refused because the derivation and its formula disagree.
pub const INVARIANCE_INCONSISTENT_DETAIL: &str = "scenario_invariance.derivation_inconsistent";
/// Detail of a report refused because the graph holds a non-static node.
pub const INVARIANCE_NON_STATIC_DETAIL: &str = "scenario_invariance.non_static_node";

/// A refused invariance report.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("{detail}: {message}")]
pub struct InvarianceReportError {
    /// Registered runtime-refusal reason code.
    pub code: &'static str,
    /// Stable `scenario_invariance.*` detail.
    pub detail: &'static str,
    /// Human-readable explanation.
    pub message: String,
}

impl InvarianceReportError {
    fn inconsistent(message: impl Into<String>) -> Self {
        Self {
            code: reason_code!("invalid_argument"),
            detail: INVARIANCE_INCONSISTENT_DETAIL,
            message: message.into(),
        }
    }

    fn non_static() -> Self {
        Self {
            code: reason_code!("invalid_argument"),
            detail: INVARIANCE_NON_STATIC_DETAIL,
            message: "a scenario graph node is not a static variable".into(),
        }
    }
}

/// What may differ between source and target under one scenario, and the graph
/// the scenario assumes. All lists are sorted by variable id.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SelectionDifferences {
    /// Variables whose mechanism may differ between the populations.
    pub selection_targets: Vec<VariableId>,
    /// Every other variable: mechanisms assumed shared between the populations.
    pub shared_mechanisms: Vec<VariableId>,
    /// Directed edges `from -> to` the scenario assumes.
    pub directed_edges: Vec<(VariableId, VariableId)>,
    /// Bidirected edges `a <-> b` (smaller id first) the scenario assumes.
    pub bidirected_edges: Vec<(VariableId, VariableId)>,
}

/// One factor the identified formula takes from a source population, and the
/// invariance that taking it assumes.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct SourceInvariance {
    /// Population the factor is measured in.
    pub population: Arc<str>,
    /// Factor variables (the district or outcome set), sorted.
    pub variables: Vec<VariableId>,
    /// Conditioning variables, sorted. Their mechanisms are NOT assumed shared
    /// (a pretreatment variable is conditioned on because it may differ).
    pub conditioned_on: Vec<VariableId>,
    /// Variables held by `do(.)` in the regime, sorted.
    pub do_set: Vec<VariableId>,
    /// Catalog regime id the bound factor is read from.
    pub regime: Option<u32>,
    /// The mechanisms assumed invariant between source and target: the factor's
    /// variables. None is a selection target.
    pub invariant_mechanisms: Vec<VariableId>,
    /// Selection targets inside the factor's bidirected district (over the
    /// nodes neither intervened nor conditioned on); empty when the district is
    /// selection-free.
    pub district_selection_targets: Vec<VariableId>,
    /// Rule of the proof step that produced the factor.
    pub rule: &'static str,
}

/// One factor the formula takes from the target population: a target law, not
/// an invariance.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct TargetFactor {
    /// Factor variables, sorted.
    pub variables: Vec<VariableId>,
    /// Conditioning variables, sorted.
    pub conditioned_on: Vec<VariableId>,
    /// Catalog regime id the bound factor is read from.
    pub regime: Option<u32>,
}

/// The rule-2 reduction of a conditional question, when the scenario answers one.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConditionalReduction {
    /// Conditioned coordinates moved into the intervention set.
    pub moves: Vec<VariableId>,
    /// Conditioned coordinates that stay conditioned on.
    pub remaining: Vec<VariableId>,
}

/// Why a scenario is not transportable: the obstruction's variables.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ObstructionWitness {
    /// `s_hedge`, `conditional_two_model_witness` or
    /// `conditional_reduced_s_hedge_candidate`.
    pub kind: &'static str,
    /// Nodes of the larger s-hedge forest, sorted (empty without a hedge).
    pub larger_nodes: Vec<VariableId>,
    /// Nodes of the smaller s-hedge forest, sorted.
    pub smaller_nodes: Vec<VariableId>,
    /// Directed edges of the larger forest, sorted.
    pub larger_directed: Vec<(VariableId, VariableId)>,
    /// Bidirected edges of the larger forest, sorted.
    pub larger_bidirected: Vec<(VariableId, VariableId)>,
    /// Directed edges of the smaller forest, sorted.
    pub smaller_directed: Vec<(VariableId, VariableId)>,
    /// Bidirected edges of the smaller forest, sorted.
    pub smaller_bidirected: Vec<(VariableId, VariableId)>,
    /// The scenario's selection targets that lie in the larger forest.
    pub selection_targets_in_larger: Vec<VariableId>,
    /// Rule-2 moves of a conditional question; empty otherwise.
    pub moves: Vec<VariableId>,
    /// Conditioned coordinates that stay conditioned on; empty otherwise.
    pub remaining: Vec<VariableId>,
}

/// The body of a report, by how the scenario was decided.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum InvarianceBody {
    /// Identified: the factors the answer relies on.
    Identified {
        /// Rules of the reachable proof steps, sorted without repeats.
        rules: Vec<&'static str>,
        /// Factors taken from a source population, sorted.
        invariances: Vec<SourceInvariance>,
        /// Factors taken from the target population, sorted.
        target_factors: Vec<TargetFactor>,
        /// The rule-2 reduction, for a conditional question.
        conditional: Option<ConditionalReduction>,
    },
    /// Proven not transportable: the obstruction, no invariance list.
    Obstructed {
        /// The obstruction's variables.
        witness: ObstructionWitness,
    },
    /// Neither identified nor proven non-transportable: no invariance list.
    Undecided {
        /// Scenario status name.
        status: &'static str,
        /// Unmet obligations or scope notes (a stop's code when unevaluated).
        obligations: Vec<Arc<str>>,
        /// An inspection-only s-hedge candidate; never an impossibility claim.
        candidate: Option<ObstructionWitness>,
    },
}

/// Selection differences and invariances of one scenario.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InvarianceReport {
    status: &'static str,
    selection: SelectionDifferences,
    body: InvarianceBody,
    identity: String,
}

impl InvarianceReport {
    /// Scenario status name (`identified`, `structurally_unidentified`, ...).
    #[must_use]
    pub const fn status(&self) -> &'static str {
        self.status
    }

    /// The selection targets, shared mechanisms and assumed edges.
    #[must_use]
    pub const fn selection(&self) -> &SelectionDifferences {
        &self.selection
    }

    /// The report body.
    #[must_use]
    pub const fn body(&self) -> &InvarianceBody {
        &self.body
    }

    /// The source factors the answer relies on; `None` unless identified.
    #[must_use]
    pub fn invariances(&self) -> Option<&[SourceInvariance]> {
        match &self.body {
            InvarianceBody::Identified { invariances, .. } => Some(invariances),
            _ => None,
        }
    }

    /// The target factors the answer uses; `None` unless identified.
    #[must_use]
    pub fn target_factors(&self) -> Option<&[TargetFactor]> {
        match &self.body {
            InvarianceBody::Identified { target_factors, .. } => Some(target_factors),
            _ => None,
        }
    }

    /// The obstruction of a proven non-transportable scenario.
    #[must_use]
    pub const fn obstruction(&self) -> Option<&ObstructionWitness> {
        match &self.body {
            InvarianceBody::Obstructed { witness } => Some(witness),
            _ => None,
        }
    }

    /// Canonical identity: a digest of the canonical text. It does not depend on
    /// the order edges or selection targets were supplied in, nor on the
    /// scenario's name.
    #[must_use]
    pub fn identity(&self) -> &str {
        &self.identity
    }

    /// The canonical text the identity digests.
    #[must_use]
    pub fn canonical_text(&self) -> String {
        canonical_text(self.status, &self.selection, &self.body)
    }
}

fn ids(values: &[VariableId]) -> String {
    values.iter().map(|v| v.raw().to_string()).collect::<Vec<_>>().join(",")
}

fn edges(values: &[(VariableId, VariableId)]) -> String {
    values.iter().map(|(a, b)| format!("{}-{}", a.raw(), b.raw())).collect::<Vec<_>>().join(",")
}

fn regime_text(regime: Option<u32>) -> String {
    regime.map_or_else(|| "none".to_owned(), |r| r.to_string())
}

fn witness_text(out: &mut String, witness: &ObstructionWitness) {
    let _ = writeln!(
        out,
        "witness={};larger={};smaller={};ld={};lb={};sd={};sb={};sel={};moves={};remaining={}",
        witness.kind,
        ids(&witness.larger_nodes),
        ids(&witness.smaller_nodes),
        edges(&witness.larger_directed),
        edges(&witness.larger_bidirected),
        edges(&witness.smaller_directed),
        edges(&witness.smaller_bidirected),
        ids(&witness.selection_targets_in_larger),
        ids(&witness.moves),
        ids(&witness.remaining),
    );
}

fn canonical_text(status: &str, selection: &SelectionDifferences, body: &InvarianceBody) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "status={status}");
    let _ = writeln!(out, "targets={}", ids(&selection.selection_targets));
    let _ = writeln!(out, "shared={}", ids(&selection.shared_mechanisms));
    let _ = writeln!(out, "directed={}", edges(&selection.directed_edges));
    let _ = writeln!(out, "bidirected={}", edges(&selection.bidirected_edges));
    match body {
        InvarianceBody::Identified { rules, invariances, target_factors, conditional } => {
            let _ = writeln!(out, "rules={}", rules.join(","));
            for item in invariances {
                let _ = writeln!(
                    out,
                    "source={};vars={};cond={};do={};regime={};district_sel={};rule={}",
                    item.population,
                    ids(&item.variables),
                    ids(&item.conditioned_on),
                    ids(&item.do_set),
                    regime_text(item.regime),
                    ids(&item.district_selection_targets),
                    item.rule,
                );
            }
            for item in target_factors {
                let _ = writeln!(
                    out,
                    "target;vars={};cond={};regime={}",
                    ids(&item.variables),
                    ids(&item.conditioned_on),
                    regime_text(item.regime),
                );
            }
            if let Some(reduction) = conditional {
                let _ = writeln!(
                    out,
                    "conditional;moves={};remaining={}",
                    ids(&reduction.moves),
                    ids(&reduction.remaining),
                );
            }
        }
        InvarianceBody::Obstructed { witness } => witness_text(&mut out, witness),
        InvarianceBody::Undecided { status, obligations, candidate } => {
            let _ = writeln!(out, "undecided={status}");
            for obligation in obligations {
                let _ = writeln!(out, "obligation={obligation}");
            }
            if let Some(witness) = candidate {
                witness_text(&mut out, witness);
            }
        }
    }
    out
}

fn digest(text: &str) -> String {
    let mut hasher = blake3::Hasher::new_derive_key("antecedent.scenario_invariance.report.v1");
    hasher.update(&(text.len() as u64).to_le_bytes());
    hasher.update(text.as_bytes());
    format!("inv.v1.{}", hasher.finalize().to_hex())
}

fn sorted(mut values: Vec<VariableId>) -> Vec<VariableId> {
    values.sort_unstable();
    values.dedup();
    values
}

fn variable_at(
    graph: &antecedent_graph::Admg,
    index: usize,
) -> Result<VariableId, InvarianceReportError> {
    match graph.nodes().get(index) {
        Some(NodeRef::Static(variable)) => Ok(*variable),
        _ => Err(InvarianceReportError::non_static()),
    }
}

fn unordered(a: VariableId, b: VariableId) -> (VariableId, VariableId) {
    if a <= b { (a, b) } else { (b, a) }
}

impl SelectionDifferences {
    /// The selection targets, shared mechanisms and edges of `diagram`.
    ///
    /// # Errors
    /// A graph node that is not a static variable.
    pub fn of(diagram: &SelectionDiagram) -> Result<Self, InvarianceReportError> {
        let graph = diagram.causal_graph();
        let mut variables = Vec::with_capacity(graph.node_count());
        let mut directed = Vec::new();
        let mut bidirected = Vec::new();
        for index in 0..graph.node_count() {
            let from = variable_at(graph, index)?;
            variables.push(from);
            let dense = DenseNodeId::from_raw(u32::try_from(index).expect("graph capacity"));
            for child in graph.children(dense) {
                directed.push((from, variable_at(graph, child.as_usize())?));
            }
            for neighbor in graph.bidirected_neighbors(dense) {
                if neighbor.as_usize() > index {
                    bidirected.push(unordered(from, variable_at(graph, neighbor.as_usize())?));
                }
            }
        }
        directed.sort_unstable();
        directed.dedup();
        bidirected.sort_unstable();
        bidirected.dedup();
        let targets = sorted(diagram.selection_targets().to_vec());
        let shared =
            sorted(variables.into_iter().filter(|v| !targets.contains(v)).collect::<Vec<_>>());
        Ok(Self {
            selection_targets: targets,
            shared_mechanisms: shared,
            directed_edges: directed,
            bidirected_edges: bidirected,
        })
    }
}

/// Selection targets inside the bidirected district of `variables`, over the
/// nodes not in `excluded`.
fn district_selection_targets(
    diagram: &SelectionDiagram,
    variables: &[VariableId],
    excluded: &BTreeSet<VariableId>,
) -> Result<Vec<VariableId>, InvarianceReportError> {
    let graph = diagram.causal_graph();
    let mut seen = BTreeSet::new();
    let mut pending = Vec::new();
    for variable in variables {
        let index =
            graph.nodes().iter().position(|node| *node == NodeRef::Static(*variable)).ok_or_else(
                || {
                    InvarianceReportError::inconsistent(format!(
                        "factor variable {} is not in the scenario graph",
                        variable.raw()
                    ))
                },
            )?;
        if !excluded.contains(variable) && seen.insert(index) {
            pending.push(index);
        }
    }
    while let Some(index) = pending.pop() {
        let dense = DenseNodeId::from_raw(u32::try_from(index).expect("graph capacity"));
        for neighbor in graph.bidirected_neighbors(dense) {
            let next = neighbor.as_usize();
            if !excluded.contains(&variable_at(graph, next)?) && seen.insert(next) {
                pending.push(next);
            }
        }
    }
    let mut found = Vec::new();
    for index in seen {
        let variable = variable_at(graph, index)?;
        if diagram.mechanism_may_differ(variable) {
            found.push(variable);
        }
    }
    Ok(sorted(found))
}

/// Leaf-producing proof steps reachable from the root step, with the leaves
/// their outputs hold.
fn producing_leaves(
    derivation: &ClassicalTransportDerivation,
) -> Vec<(&'static str, LeafSignature)> {
    let mut used = vec![false; derivation.proof.len()];
    let mut pending = vec![derivation.root_step];
    while let Some(step) = pending.pop() {
        if step >= used.len() || used[step] {
            continue;
        }
        used[step] = true;
        pending.extend(derivation.proof[step].children.iter().copied());
    }
    let mut out = Vec::new();
    for (step, reachable) in derivation.proof.iter().zip(used) {
        if reachable
            && matches!(step.rule, Rule::Source | Rule::DirectTransport | Rule::Standardize)
        {
            for leaf in derivation.arena.leaf_signatures(step.output) {
                out.push((step.rule.name(), leaf));
            }
        }
    }
    out
}

fn same_factor(a: &LeafSignature, b: &LeafSignature) -> bool {
    a.variables == b.variables
        && a.conditioned_on == b.conditioned_on
        && a.intervention == b.intervention
        && a.domain == b.domain
        && a.binding.population == b.binding.population
}

fn identified_body(
    diagram: &SelectionDiagram,
    bound: &BoundTransportFunctional,
    conditional: Option<ConditionalReduction>,
) -> Result<InvarianceBody, InvarianceReportError> {
    let derivation = bound.derivation();
    let formula = derivation.arena.leaf_signatures(derivation.root);
    let producing = producing_leaves(derivation);
    let bound_leaves = bound.leaf_factors();
    let regime_of = |leaf: &LeafSignature| {
        bound_leaves
            .iter()
            .find(|(_, candidate)| same_factor(candidate, leaf))
            .and_then(|(_, candidate)| candidate.binding.regime)
            .map(antecedent_core::RegimeId::raw)
    };
    let mut invariances = Vec::new();
    let mut target_factors = Vec::new();
    for leaf in &formula {
        let variables = sorted(leaf.variables.to_vec());
        let conditioned_on = sorted(leaf.conditioned_on.to_vec());
        if leaf.binding.population == derivation.query.target {
            target_factors.push(TargetFactor {
                variables,
                conditioned_on,
                regime: regime_of(leaf),
            });
            continue;
        }
        let rule = producing
            .iter()
            .find(|(_, produced)| produced == leaf)
            .map(|(rule, _)| *rule)
            .ok_or_else(|| {
            InvarianceReportError::inconsistent(
                "a source factor of the formula is produced by no reachable proof step",
            )
        })?;
        if let Some(target) = variables.iter().find(|v| diagram.mechanism_may_differ(**v)) {
            return Err(InvarianceReportError::inconsistent(format!(
                "source factor over selection target {} cannot be an invariance",
                target.raw()
            )));
        }
        let do_set = sorted(leaf.intervention.iter().map(|a| a.variable).collect());
        let excluded = do_set.iter().chain(&conditioned_on).copied().collect::<BTreeSet<_>>();
        invariances.push(SourceInvariance {
            population: Arc::clone(&leaf.binding.population),
            district_selection_targets: district_selection_targets(diagram, &variables, &excluded)?,
            invariant_mechanisms: variables.clone(),
            variables,
            conditioned_on,
            do_set,
            regime: regime_of(leaf),
            rule,
        });
    }
    invariances.sort();
    invariances.dedup();
    target_factors.sort();
    target_factors.dedup();
    let mut rules = derivation.rules();
    rules.sort_unstable();
    rules.dedup();
    Ok(InvarianceBody::Identified { rules, invariances, target_factors, conditional })
}

fn witness_of(
    kind: &'static str,
    diagram: &SelectionDiagram,
    hedge: Option<&SHedgeRecord>,
    moves: &[VariableId],
    remaining: &[VariableId],
) -> ObstructionWitness {
    let nodes = |raw: &[u32]| sorted(raw.iter().copied().map(VariableId::from_raw).collect());
    let pairs = |raw: &[(u32, u32)]| {
        let mut out = raw
            .iter()
            .map(|(a, b)| (VariableId::from_raw(*a), VariableId::from_raw(*b)))
            .collect::<Vec<_>>();
        out.sort_unstable();
        out.dedup();
        out
    };
    let bidirected = |raw: &[(u32, u32)]| {
        let mut out = pairs(raw).into_iter().map(|(a, b)| unordered(a, b)).collect::<Vec<_>>();
        out.sort_unstable();
        out.dedup();
        out
    };
    let larger_nodes = hedge.map(|h| nodes(&h.larger.nodes)).unwrap_or_default();
    let selection_targets_in_larger = sorted(
        diagram
            .selection_targets()
            .iter()
            .copied()
            .filter(|target| larger_nodes.contains(target))
            .collect(),
    );
    ObstructionWitness {
        kind,
        smaller_nodes: hedge.map(|h| nodes(&h.smaller.nodes)).unwrap_or_default(),
        larger_directed: hedge.map(|h| pairs(&h.larger.directed)).unwrap_or_default(),
        larger_bidirected: hedge.map(|h| bidirected(&h.larger.bidirected)).unwrap_or_default(),
        smaller_directed: hedge.map(|h| pairs(&h.smaller.directed)).unwrap_or_default(),
        smaller_bidirected: hedge.map(|h| bidirected(&h.smaller.bidirected)).unwrap_or_default(),
        larger_nodes,
        selection_targets_in_larger,
        moves: moves.to_vec(),
        remaining: remaining.to_vec(),
    }
}

/// The selection differences and invariances (or structural obstruction) of one
/// decided scenario.
///
/// # Errors
/// A graph with a non-static node, or a derivation whose formula holds a source
/// factor that no reachable proof step produced (or one over a selection
/// target): a contradiction a checked derivation cannot contain.
pub fn invariance_report(
    scenario: &TransportScenario,
    outcome: &ScenarioOutcome,
) -> Result<InvarianceReport, InvarianceReportError> {
    let diagram = &scenario.diagram;
    let selection = SelectionDifferences::of(diagram)?;
    let body = match outcome {
        ScenarioOutcome::Identified(bound) => identified_body(diagram, bound, None)?,
        ScenarioOutcome::ConditionalIdentified(bound) => {
            let derivation = bound.derivation();
            let reduction = ConditionalReduction {
                moves: derivation.moves().to_vec(),
                remaining: derivation.remaining().to_vec(),
            };
            identified_body(diagram, bound.joint(), Some(reduction))?
        }
        ScenarioOutcome::StructurallyUnidentified(hedge) => InvarianceBody::Obstructed {
            witness: witness_of("s_hedge", diagram, Some(hedge.as_ref()), &[], &[]),
        },
        ScenarioOutcome::ConditionalProvenNonTransportable(proof) => {
            let hedge = proof.candidate().map(|c| c.s_hedge().to_record());
            InvarianceBody::Obstructed {
                witness: witness_of(
                    "conditional_two_model_witness",
                    diagram,
                    hedge.as_ref(),
                    proof.moves(),
                    proof.remaining(),
                ),
            }
        }
        ScenarioOutcome::MissingEvidence { obligations }
        | ScenarioOutcome::NotCertified { obligations } => InvarianceBody::Undecided {
            status: outcome.status(),
            obligations: obligations.to_vec(),
            candidate: None,
        },
        ScenarioOutcome::ConditionalNotCertified { obligations, candidate } => {
            InvarianceBody::Undecided {
                status: outcome.status(),
                obligations: obligations.to_vec(),
                candidate: candidate.as_ref().map(|c| {
                    witness_of(
                        "conditional_reduced_s_hedge_candidate",
                        diagram,
                        Some(&c.s_hedge().to_record()),
                        c.moves(),
                        c.remaining(),
                    )
                }),
            }
        }
        ScenarioOutcome::Unevaluated { stop } => InvarianceBody::Undecided {
            status: outcome.status(),
            obligations: vec![Arc::from(stop.code())],
            candidate: None,
        },
    };
    let status = outcome.status();
    let identity = digest(&canonical_text(status, &selection, &body));
    Ok(InvarianceReport { status, selection, body, identity })
}
