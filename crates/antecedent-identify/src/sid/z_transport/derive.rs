//! Derive items of the parent module.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::{IdentificationBudget, IdentificationError};
use antecedent_core::VariableId;
use antecedent_expr::{CausalExprArena, DomainRef, ExprId, ExprNode};
use antecedent_graph::{BitSet, DenseNodeId, NodeRef, SelectionDiagram};
use std::sync::Arc;
use super::search::z_search;
use super::search::{matches_registered_surrogate_graph, surrogate_formula};

use super::*;

/// [`verify_z_transport_obstruction`] with its replay charged to `meter`.
pub(super) fn verify_z_obstruction_metered(
    diagram: &SelectionDiagram,
    query: &ZTransportQuery,
    obstruction: &ZTransportObstruction,
    meter: super::super::SidMeter<'_>,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<(), IdentificationError> {
    validate_z_transport_query(diagram, query)?;
    if obstruction.query != *query || obstruction.graph_signature != super::super::graph_signature(diagram)
    {
        return Err(IdentificationError::invalid_derivation(
            "z_transport.obstruction_input_mismatch",
        ));
    }
    verify_trz_line11_terminal(diagram, query, &obstruction.terminal, ctx)?;
    let assignment = query
        .experiment_assignment
        .iter()
        .filter_map(|intervention| {
            intervention.value.as_f64().map(|value| (intervention.variable.raw(), value))
        })
        .collect::<Vec<_>>();
    if assignment != obstruction.selected_assignment {
        return Err(IdentificationError::invalid_derivation(
            "z_transport.obstruction_assignment_mismatch",
        ));
    }
    let replay = z_search(diagram, query, meter, ctx, &mut None)?;
    if replay.identified.is_some()
        || replay.terminal_failure.as_ref() != Some(&obstruction.terminal)
    {
        return Err(IdentificationError::invalid_derivation(
            "z_transport.obstruction_replay_mismatch",
        ));
    }
    Ok(())
}

/// Check the local graph premises of Bareinboim–Pearl `TRz` Figure 4 line 11
/// independently of the recursive search's terminal flag and separation
/// helper. Their Theorem 5 maps this failed line-11 state `(D, C0)` to a
/// zs-hedge; `verify_z_transport_obstruction` separately replays the prefix.
fn verify_trz_line11_terminal(
    diagram: &SelectionDiagram,
    query: &ZTransportQuery,
    terminal: &TrzTerminalFailure,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<(), IdentificationError> {
    let bad = || IdentificationError::invalid_derivation("z_transport.invalid_line11_terminal");
    let classical_query = super::super::ClassicalTransportQuery {
        outcomes: Arc::clone(&query.outcomes),
        treatments: Arc::clone(&query.treatments),
        source: Arc::clone(&query.source),
        target: Arc::clone(&query.target),
    };
    let engine = super::super::Engine::new(diagram, &classical_query, super::super::SidLimits::default(), ctx)?;
    let set = |raws: &[u32]| -> Result<BitSet, IdentificationError> {
        let variables = raws.iter().copied().map(VariableId::from_raw).collect::<Vec<_>>();
        if variables.iter().copied().collect::<std::collections::BTreeSet<_>>().len()
            != variables.len()
        {
            return Err(bad());
        }
        engine.set(&variables).map_err(|_| bad())
    };
    let y = set(&terminal.outcomes)?;
    let x = set(&terminal.treatments)?;
    let v = set(&terminal.vertices)?;
    if !y.any()
        || !x.any()
        || y.iter().any(|node| !v.contains(node))
        || x.iter().any(|node| !v.contains(node))
    {
        return Err(bad());
    }
    let mut candidate_active = Vec::new();
    for raw in &terminal.remaining_controllable {
        let variable = VariableId::from_raw(*raw);
        let dense = engine.prepared.var_to_dense(variable).map_err(|_| bad())?;
        if x.contains(dense) {
            candidate_active.push(*raw);
        }
    }
    candidate_active.sort_unstable();
    if candidate_active != terminal.candidate_active {
        return Err(bad());
    }
    let c0 = super::super::difference(&v, &x);
    let districts = engine.prepared.c_components(&c0);
    if districts.len() != 1 || engine.prepared.c_components(&v).len() != 1 {
        return Err(bad());
    }
    let mut c0_variables = engine.vars(&districts[0])?.iter().map(|v| v.raw()).collect::<Vec<_>>();
    c0_variables.sort_unstable();
    if c0_variables != terminal.c0 {
        return Err(bad());
    }
    let state = super::super::State { y, x, v, kernel: ExprId::from_raw(0) };
    let separation = engine.independently_admissible(&state, &[], diagram.selection_targets())?;
    if separation != terminal.selection_separated
        || (!candidate_active.is_empty() && separation)
        || terminal.rules.last().map(String::as_str) != Some("ztr.line11.fail")
    {
        return Err(bad());
    }
    Ok(())
}

/// Identify a bounded single-source restricted-experiment query under the
/// caller's search limits and execution context.
///
/// The registered surrogate formula and direct joint exchange have dedicated
/// local checkers. Other in-bound graphs, including admissible selection
/// diagrams, follow the recursive `TRz` reduction and are rederived on replay.
/// Use [`decide_z_transport_with_catalog`] when a negative result must be
/// distinguished from incomplete catalog evidence: a line-11 failure without a
/// catalog is [`ZTransportResult::NotCertified`], not an obstruction.
///
/// The registered factorization is the four-variable graph `W→Z→X→Y`, `W→Y`,
/// bidirected `W↔Y`, `Z↔Y`, `Z↔X`, controllable `Z`, query `P(Y | do(X))`, and
/// no selection targets. Other graphs up to 12 observed and 4 controllable
/// variables publish the recursive derivation when search succeeds.
///
/// # Errors
/// Invalid query coordinates, cancellation, or exhausted computation.
pub fn identify_z_transport(
    diagram: &SelectionDiagram,
    query: &ZTransportQuery,
    limits: super::super::SidLimits,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<ZTransportResult, IdentificationError> {
    identify_z_transport_reporting(diagram, query, limits, ctx, &mut None)
}

/// Identify a bounded z-transport formula, and on a budget or cancellation fill
/// `receipt` with the limits in force and what the search consumed before
/// returning the exhaustion error, so callers that snapshot a failure can
/// expose a limits receipt instead of an opaque status.
///
/// # Errors
/// Invalid query coordinates, cancellation, or exhausted computation.
pub fn identify_z_transport_reporting(
    diagram: &SelectionDiagram,
    query: &ZTransportQuery,
    limits: super::super::SidLimits,
    ctx: &antecedent_core::ExecutionContext,
    receipt: &mut Option<ZTransportLimitsReceipt>,
) -> Result<ZTransportResult, IdentificationError> {
    Ok(match derive_z_transport(diagram, query, super::super::SidMeter::Limits(limits), ctx, receipt)? {
        ZDerivation::Formula(derivation) => ZTransportResult::Identified(Box::new(derivation)),
        ZDerivation::Unassigned { inspection, .. } => ZTransportResult::NotCertified {
            reason: "z_transport.experiment_assignment_required",
            inspection,
        },
        ZDerivation::Line11 { inspection, .. } | ZDerivation::NotCertified { inspection, .. } => {
            ZTransportResult::NotCertified {
                reason: "z_transport.no_checked_recursive_formula",
                inspection,
            }
        }
    })
}

#[allow(
    clippy::large_enum_variant,
    reason = "Private short-lived state; boxing would add allocation to recursive search"
)]
pub(super) enum ZDerivation {
    Formula(ZTransportDerivation),
    Line11 { terminal: TrzTerminalFailure, inspection: ZTransportNotCertifiedInspection },
    Unassigned { variable: VariableId, inspection: ZTransportNotCertifiedInspection },
    NotCertified { reason: &'static str, inspection: ZTransportNotCertifiedInspection },
}

pub(super) fn derive_z_transport(
    diagram: &SelectionDiagram,
    query: &ZTransportQuery,
    meter: super::super::SidMeter<'_>,
    ctx: &antecedent_core::ExecutionContext,
    receipt: &mut Option<ZTransportLimitsReceipt>,
) -> Result<ZDerivation, IdentificationError> {
    validate_z_transport_query(diagram, query)?;
    let limits = meter.limits();
    if limits.steps == 0 || limits.depth == 0 {
        *receipt = Some(ZTransportLimitsReceipt {
            budget: if limits.depth == 0 {
                ZTransportBudgetKind::Depth
            } else {
                ZTransportBudgetKind::Steps
            },
            steps_limit: limits.steps,
            depth_limit: limits.depth,
            steps_consumed: Some(0),
            depth_reached: Some(0),
            explored_rules: Vec::new(),
        });
        return Err(IdentificationError::budget(IdentificationBudget::ZTransport));
    }
    if ctx.cancellation.is_cancelled() {
        *receipt = Some(ZTransportLimitsReceipt {
            budget: ZTransportBudgetKind::Cancelled,
            steps_limit: limits.steps,
            depth_limit: limits.depth,
            steps_consumed: Some(0),
            depth_reached: Some(0),
            explored_rules: Vec::new(),
        });
        return Err(IdentificationError::Cancelled);
    }
    if direct_joint_admissible(diagram, query)? {
        let (arena, root) = direct_joint_formula(query)?;
        let derivation = ZTransportDerivation {
            query: query.clone(),
            graph_signature: super::super::graph_signature(diagram),
            surrogate: Some(query.experiment_assignment[0].variable),
            confounder: None,
            arena,
            root,
            kind: ZFormulaKind::DirectJoint,
            rules: vec!["ztr.source_exchange_joint".into()],
        };
        verify_z_transport_derivation(diagram, query, &derivation, limits, ctx)?;
        return Ok(ZDerivation::Formula(derivation));
    }
    if let Some(derivation) = registered_surrogate_derivation(diagram, query, limits, ctx)? {
        return Ok(ZDerivation::Formula(derivation));
    }
    let searched = z_search(diagram, query, meter, ctx, receipt)?;
    let explored_rules = searched.explored_rules.clone();
    let steps_explored = searched.steps_explored;
    let depth_reached = searched.depth_reached;
    let inspection = |kind: ZTransportNotCertifiedKind| ZTransportNotCertifiedInspection {
        kind,
        explored_rules: explored_rules.clone(),
        steps_explored,
        depth_reached,
    };
    if let Some(variable) = searched.unassigned {
        return Ok(ZDerivation::Unassigned {
            variable,
            inspection: inspection(ZTransportNotCertifiedKind::ExperimentAssignmentRequired),
        });
    }
    if let Some((arena, root, trace)) = searched.identified {
        // The recursive formula is the search's own output: re-running the
        // search would recheck nothing. Replay verifies untrusted records.
        return Ok(ZDerivation::Formula(ZTransportDerivation {
            query: query.clone(),
            graph_signature: super::super::graph_signature(diagram),
            surrogate: None,
            confounder: None,
            arena,
            root,
            kind: ZFormulaKind::Recursive,
            rules: std::iter::once("ztr.recursive_reduction".to_owned()).chain(trace).collect(),
        }));
    }
    if let Some(terminal) = searched.terminal_failure {
        return Ok(ZDerivation::Line11 {
            terminal,
            inspection: inspection(ZTransportNotCertifiedKind::CheckedLine11Terminal),
        });
    }
    let kind = if explored_rules.iter().any(|rule| rule == "ztr.line10.c_factor_not_identified") {
        ZTransportNotCertifiedKind::SourceCFactorUnidentified
    } else {
        ZTransportNotCertifiedKind::NoRecursiveFormula
    };
    Ok(ZDerivation::NotCertified {
        reason: "z_transport.no_checked_recursive_formula",
        inspection: inspection(kind),
    })
}

fn registered_surrogate_derivation(
    diagram: &SelectionDiagram,
    query: &ZTransportQuery,
    limits: super::super::SidLimits,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<Option<ZTransportDerivation>, IdentificationError> {
    if diagram.causal_graph().node_count() != 4
        || !diagram.selection_targets().is_empty()
        || query.outcomes.len() != 1
        || query.treatments.len() != 1
        || query.controllable.len() != 1
        || query.experiment_assignment.len() != 1
        || query.experiment_assignment[0].variable != query.controllable[0]
    {
        return Ok(None);
    }
    let outcome = query.outcomes[0];
    let treatment = query.treatments[0];
    let surrogate = query.controllable[0];
    let Some(confounder) = diagram
        .causal_graph()
        .nodes()
        .iter()
        .filter_map(|node| match node {
            NodeRef::Static(v) => Some(*v),
            _ => None,
        })
        .find(|v| *v != outcome && *v != treatment && *v != surrogate)
    else {
        return Ok(None);
    };
    if !matches_registered_surrogate_graph(diagram, confounder, surrogate, treatment, outcome) {
        return Ok(None);
    }
    let (arena, root) = surrogate_formula(query, confounder)?;
    let derivation = ZTransportDerivation {
        query: query.clone(),
        graph_signature: super::super::graph_signature(diagram),
        surrogate: Some(surrogate),
        confounder: Some(confounder),
        arena,
        root,
        kind: ZFormulaKind::Surrogate,
        rules: vec!["ztr.surrogate_factorization".into()],
    };
    verify_z_transport_derivation(diagram, query, &derivation, limits, ctx)?;
    Ok(Some(derivation))
}

// A direct `TRz` source exchange is legal when the requested intervention is a
// concrete subset of the controllable family and the population selectors are
// separated from the outcome in the graph with incoming treatment arrows cut.
// The source experiment must still be bound to real evidence by the caller.
fn direct_joint_admissible(
    diagram: &SelectionDiagram,
    query: &ZTransportQuery,
) -> Result<bool, IdentificationError> {
    if query.experiment_assignment.len() != query.treatments.len()
        || !query.treatments.iter().all(|x| {
            query.controllable.contains(x)
                && query.experiment_assignment.iter().any(|a| a.variable == *x)
        })
    {
        return Ok(false);
    }
    let graph = diagram.causal_graph();
    let dense = |v: VariableId| {
        graph
            .nodes()
            .iter()
            .position(|node| *node == NodeRef::Static(v))
            .map(|i| DenseNodeId::from_raw(index_u32(i)))
            .ok_or_else(|| IdentificationError::invalid_input("z_transport.unknown_variable"))
    };
    let mut all = BitSet::with_len(graph.node_count());
    for index in 0..graph.node_count() {
        all.insert(DenseNodeId::from_raw(index_u32(index)));
    }
    let mut intervened = BitSet::with_len(graph.node_count());
    for x in query.treatments.iter().copied() {
        intervened.insert(dense(x)?);
    }
    let selectors =
        diagram.selection_targets().iter().copied().map(dense).collect::<Result<Vec<_>, _>>()?;
    let outcomes = query.outcomes.iter().copied().map(dense).collect::<Result<Vec<_>, _>>()?;
    let graph = super::super::MutilatedSelection::build(graph, &all, &intervened, &selectors)?;
    graph.separates(
        &outcomes,
        &intervened.to_dense_ids(),
        &mut antecedent_graph::DSeparationWorkspace::default(),
    )
}

fn direct_joint_formula(
    query: &ZTransportQuery,
) -> Result<(CausalExprArena, ExprId), IdentificationError> {
    let mut arena = CausalExprArena::new();
    let variables = arena.intern_var_set(query.outcomes.iter().copied());
    let conditioned_on = arena.empty_var_set();
    let intervention = crate::intervention_support::intern_assignments(
        &mut arena,
        query.experiment_assignment.iter().map(|a| {
            antecedent_expr::InterventionAssignment::concrete(a.variable, a.value.clone())
        }),
    )?;
    let population = arena.intern_population(Arc::clone(&query.source));
    let root = arena.intern(ExprNode::Distribution {
        variables,
        conditioned_on,
        intervention,
        domain: DomainRef::Interventional,
        population,
        regime: None,
    });
    Ok((arena, root))
}

/// Independently recheck the graph and formula premises of a zTR derivation
/// under the caller's search limits and execution context.
///
/// # Errors
/// A changed graph, query, or formula; an exhausted replay budget; or
/// cancellation.
pub fn verify_z_transport_derivation(
    diagram: &SelectionDiagram,
    query: &ZTransportQuery,
    derivation: &ZTransportDerivation,
    limits: super::super::SidLimits,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<(), IdentificationError> {
    derivation.check_inputs(diagram, query)?;
    let input_mismatch =
        || IdentificationError::invalid_derivation("z_transport.proof_input_mismatch");
    let formula_mismatch =
        || IdentificationError::invalid_derivation("z_transport.proof_formula_mismatch");
    match derivation.kind {
        ZFormulaKind::Recursive => {
            if derivation.surrogate.is_some() || derivation.confounder.is_some() {
                return Err(input_mismatch());
            }
            let Some((expected, root, trace)) =
                z_search(diagram, query, super::super::SidMeter::Limits(limits), ctx, &mut None)?
                    .identified
            else {
                return Err(IdentificationError::invalid_derivation(
                    "z_transport.proof_rule_mismatch",
                ));
            };
            let expected_rules = std::iter::once("ztr.recursive_reduction".to_owned())
                .chain(trace)
                .collect::<Vec<_>>();
            if derivation.root != root
                || derivation.arena != expected
                || derivation.rules != expected_rules
            {
                return Err(formula_mismatch());
            }
        }
        ZFormulaKind::DirectJoint => {
            if !direct_joint_admissible(diagram, query)?
                || derivation.surrogate != Some(query.experiment_assignment[0].variable)
                || derivation.confounder.is_some()
                || derivation.rules != ["ztr.source_exchange_joint"]
            {
                return Err(input_mismatch());
            }
            let (expected, root) = direct_joint_formula(query)?;
            if derivation.root != root || derivation.arena != expected {
                return Err(formula_mismatch());
            }
        }
        ZFormulaKind::Surrogate => {
            let (Some(surrogate), Some(confounder)) = (derivation.surrogate, derivation.confounder)
            else {
                return Err(input_mismatch());
            };
            if diagram.causal_graph().node_count() != 4
                || !diagram.selection_targets().is_empty()
                || query.outcomes.len() != 1
                || query.treatments.len() != 1
                || query.controllable.as_ref() != [surrogate]
                || query.experiment_assignment.len() != 1
                || query.experiment_assignment[0].variable != surrogate
                || derivation.rules != ["ztr.surrogate_factorization"]
            {
                return Err(input_mismatch());
            }
            if !matches_registered_surrogate_graph(
                diagram,
                confounder,
                surrogate,
                query.treatments[0],
                query.outcomes[0],
            ) {
                return Err(IdentificationError::invalid_derivation(
                    "z_transport.proof_graph_mismatch",
                ));
            }
            let (expected, root) = surrogate_formula(query, confounder)?;
            if derivation.root != root || derivation.arena != expected {
                return Err(formula_mismatch());
            }
        }
    }
    Ok(())
}
