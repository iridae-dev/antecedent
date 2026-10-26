//! Decide items of the parent module.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::IdentificationError;
use antecedent_core::{EvidenceCatalog, InterventionAssignment as CatalogInterventionAssignment, VariableId};
use antecedent_graph::{BitSet, NodeRef, SelectionDiagram};
use std::sync::Arc;
use super::search::{pre_search_budget_kind, same_variable_set};

use super::*;

/// Decide a bounded single-source z-transport query against the supplied catalog.
///
/// A positive result is a checked `TRz` formula whose cited joints are present.
/// Unused experiments and uncited variables are not required. A negative result
/// is certified when search reaches `TRz` line 11 and an independent checker
/// confirms the reduced graph, C0, Z∩X, and the failed line-10 separation
/// premise. That obstruction is a fact about the graph and the declared
/// controllable set; an incomplete catalog is not the obstruction. A missing
/// cited factor, or a line-10 exchange whose controllable has no declared
/// level, remains missing evidence. Exhausted search remains distinct from
/// theorem obstruction.
///
/// # Errors
/// Invalid or unsupported query, cancellation, or exhausted search.
pub fn decide_z_transport_with_catalog(
    diagram: &SelectionDiagram,
    query: &ZTransportQuery,
    catalog: &EvidenceCatalog,
    limits: super::super::SidLimits,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<ZTransportDecision, IdentificationError> {
    let meter = super::super::SidMeter::Limits(limits);
    decide_z_transport_reporting(diagram, query, catalog, meter, ctx, &mut None)
}

/// Decide a bounded single-source z-transport query, and on a budget or
/// cancellation return the limits receipt rather than an opaque exhaustion
/// error, so callers can inspect the limits in force and what was consumed.
///
/// Every non-exhaustion outcome is a [`ZTransportOutcome::Decided`]; only a
/// budget or cooperative cancellation becomes [`ZTransportOutcome::Exhausted`].
/// Malformed inputs and invalid catalogs are still returned as errors.
///
/// # Errors
/// Invalid or unsupported query, or an invalid catalog.
pub fn decide_z_transport_inspecting(
    diagram: &SelectionDiagram,
    query: &ZTransportQuery,
    catalog: &EvidenceCatalog,
    limits: super::super::SidLimits,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<ZTransportOutcome, IdentificationError> {
    let mut receipt = None;
    let meter = super::super::SidMeter::Limits(limits);
    match decide_z_transport_reporting(diagram, query, catalog, meter, ctx, &mut receipt) {
        Ok(decision) => Ok(ZTransportOutcome::Decided(decision)),
        Err(error) if error.is_budget_or_cancel() => {
            let receipt = receipt.unwrap_or_else(|| ZTransportLimitsReceipt {
                budget: pre_search_budget_kind(&error),
                steps_limit: limits.steps,
                depth_limit: limits.depth,
                steps_consumed: None,
                depth_reached: None,
                explored_rules: Vec::new(),
            });
            Ok(ZTransportOutcome::Exhausted(receipt))
        }
        Err(error) => Err(error),
    }
}

/// Decide a single-source query charging `meter`: its own limits for the
/// public entry points, or the shared budget of an enclosing mz decision.
pub(crate) fn decide_z_transport_reporting(
    diagram: &SelectionDiagram,
    query: &ZTransportQuery,
    catalog: &EvidenceCatalog,
    mut meter: super::super::SidMeter<'_>,
    ctx: &antecedent_core::ExecutionContext,
    receipt: &mut Option<ZTransportLimitsReceipt>,
) -> Result<ZTransportDecision, IdentificationError> {
    catalog.validate().map_err(|error| {
        IdentificationError::invalid_catalog(format!("z_transport.invalid_catalog: {error}"))
    })?;
    match derive_z_transport(diagram, query, meter.reborrow(), ctx, receipt)? {
        ZDerivation::Formula(derivation) => {
            match bind_z_transport_catalog(diagram, derivation.query(), &derivation, catalog) {
                Ok(_) => Ok(ZTransportDecision::Identified(Box::new(derivation))),
                Err(error @ IdentificationError::MissingEvidence { .. }) => {
                    Ok(ZTransportDecision::MissingEvidence {
                        missing: ZTransportMissingEvidence::CitedFactor {
                            detail: error.to_string(),
                        },
                    })
                }
                Err(error) => Err(error),
            }
        }
        ZDerivation::Unassigned { variable, .. } => Ok(ZTransportDecision::MissingEvidence {
            missing: ZTransportMissingEvidence::UnassignedControllable { variable },
        }),
        ZDerivation::Line11 { terminal, .. } => {
            certify_line11_obstruction(diagram, query, terminal, meter, ctx)
        }
        ZDerivation::NotCertified { reason, inspection } => {
            Ok(ZTransportDecision::NotCertified { reason, inspection })
        }
    }
}

/// Search two sources separately on a shared causal graph.
///
/// One identifying source is enough, and the other source's catalog is not
/// required: both sources are searched, and a failure of one source does not
/// hide an identification by the other. Two line-11 terminals become one
/// obstruction only for one experiment family. A result that would mix a
/// factor from each source is refused by name. This does not call classical
/// meta-transport, which assumes every source can experiment on every variable.
///
/// # Errors
/// A source population collides with the other source or the target, a
/// selection diagram is invalid, or neither source identifies and one search
/// failed (the first failure is reported).
#[allow(
    clippy::too_many_lines,
    reason = "Mirrors the single-source, disconnected, and connected branches explicitly"
)]
pub fn decide_two_source_z_transport(
    graph: &antecedent_graph::Admg,
    query: &TwoSourceZTransportQuery,
    catalogs: [&EvidenceCatalog; 2],
    limits: super::super::SidLimits,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<TwoSourceZTransportDecision, IdentificationError> {
    if query.sources[0].population == query.sources[1].population
        || query.sources.iter().any(|source| source.population == query.target)
    {
        return Err(IdentificationError::invalid_input(
            "z_transport.two_source_population_collision",
        ));
    }
    let mut decisions = Vec::with_capacity(2);
    for (source, catalog) in query.sources.iter().zip(catalogs) {
        let diagram =
            SelectionDiagram::try_new(graph.clone(), Arc::clone(&source.selection_targets))
                .map_err(|error| IdentificationError::invalid_input(error.to_string()))?;
        let single = ZTransportQuery {
            outcomes: Arc::clone(&query.outcomes),
            treatments: Arc::clone(&query.treatments),
            controllable: Arc::clone(&source.controllable),
            experiment_assignment: Arc::clone(&source.experiment_assignment),
            source: Arc::clone(&source.population),
            target: Arc::clone(&query.target),
        };
        decisions.push(decide_z_transport_with_catalog(&diagram, &single, catalog, limits, ctx));
    }
    for (source, decision) in query.sources.iter().zip(&decisions) {
        if let Ok(ZTransportDecision::Identified(derivation)) = decision {
            return Ok(TwoSourceZTransportDecision::Identified {
                source: Arc::clone(&source.population),
                derivation: derivation.clone(),
            });
        }
    }
    if let Some(graph_components) = two_disconnected_query_components(graph, query) {
        for source_order in [[0usize, 1usize], [1usize, 0usize]] {
            let mut component_proofs = Vec::with_capacity(2);
            let mut complete = true;
            for (component_index, source_index) in source_order.into_iter().enumerate() {
                let nodes = &graph_components[component_index];
                let source = &query.sources[source_index];
                let outcomes = query
                    .outcomes
                    .iter()
                    .copied()
                    .filter(|variable| nodes.contains(variable))
                    .collect::<Vec<_>>();
                let treatments = query
                    .treatments
                    .iter()
                    .copied()
                    .filter(|variable| nodes.contains(variable))
                    .collect::<Vec<_>>();
                let assignment = source
                    .experiment_assignment
                    .iter()
                    .filter(|assignment| nodes.contains(&assignment.variable))
                    .cloned()
                    .collect::<Vec<_>>();
                if outcomes.is_empty()
                    || treatments.is_empty()
                    || treatments.iter().any(|variable| !source.controllable.contains(variable))
                    || !assignment.iter().all(|a| treatments.contains(&a.variable))
                    || !treatments.iter().all(|variable| {
                        assignment.iter().any(|assignment| assignment.variable == *variable)
                    })
                    || source.selection_targets.iter().any(|variable| !nodes.contains(variable))
                    || source.controllable.iter().any(|variable| !nodes.contains(variable))
                {
                    complete = false;
                    break;
                }
                let component_query = ZTransportQuery {
                    outcomes: outcomes.clone().into(),
                    treatments: treatments.clone().into(),
                    controllable: source.controllable.clone(),
                    experiment_assignment: assignment.into(),
                    source: Arc::clone(&source.population),
                    target: Arc::clone(&query.target),
                };
                let diagram =
                    SelectionDiagram::try_new(graph.clone(), Arc::clone(&source.selection_targets))
                        .map_err(|error| IdentificationError::invalid_input(error.to_string()))?;
                let ZTransportDecision::Identified(derivation) = decide_z_transport_with_catalog(
                    &diagram,
                    &component_query,
                    catalogs[source_index],
                    limits,
                    ctx,
                )?
                else {
                    complete = false;
                    break;
                };
                component_proofs.push(TwoSourceZTransportComponent {
                    source: Arc::clone(&source.population),
                    outcomes: outcomes.into(),
                    treatments: treatments.into(),
                    derivation,
                });
            }
            if complete {
                let [first, second] = component_proofs.try_into().expect("two source components");
                return Ok(TwoSourceZTransportDecision::CombinedIdentified {
                    components: [first, second],
                    factorization: ComponentFactorization::DisconnectedGraphComponents,
                });
            }
        }
    }
    // Connected complementary case: one connected graph whose intervened
    // outcomes m-separate into two groups given the treatments. Each group's
    // marginal interventional effect is transported from one source, and the
    // target law is their product. A single connected c-factor that would need
    // a fabricated joint over both sources' interventions is never combined.
    if let Some(decision) = connected_complementary_decision(graph, query, catalogs, limits, ctx)? {
        return Ok(decision);
    }
    let decisions = decisions.into_iter().collect::<Result<Vec<_>, _>>()?;
    // A line-11 terminal is an obstruction relative to one source's experiment
    // family. Two such terminals rule out the union of the families only when
    // the union is one of them: the sources declare the same controllable set
    // on the same selection diagram. Otherwise a factor from each source might
    // still combine, which this decision does not search.
    if let [
        ZTransportDecision::ProvenNonTransportable(left),
        ZTransportDecision::ProvenNonTransportable(right),
    ] = decisions.as_slice()
    {
        let [first, second] = &query.sources;
        let same_family = same_variable_set(&first.controllable, &second.controllable)
            && same_variable_set(&first.selection_targets, &second.selection_targets);
        if same_family {
            return Ok(TwoSourceZTransportDecision::ProvenNonTransportable {
                obstructions: [*left.clone(), *right.clone()],
            });
        }
    }
    Ok(TwoSourceZTransportDecision::NotCertified {
        reason: "z_transport.multi_source_combination_not_searched",
    })
}

/// Return the two graph components only for a query whose outcomes and
/// treatments partition into exactly two disconnected static components.
fn two_disconnected_query_components(
    graph: &antecedent_graph::Admg,
    query: &TwoSourceZTransportQuery,
) -> Option<[Vec<VariableId>; 2]> {
    if graph.nodes().iter().any(|node| !matches!(node, NodeRef::Static(_))) {
        return None;
    }
    let components = graph
        .weak_components()
        .into_iter()
        .map(|component| {
            component
                .into_iter()
                .map(|node| graph.static_variable(node))
                .collect::<Option<Vec<_>>>()
        })
        .collect::<Option<Vec<_>>>()?;
    if components.len() != 2
        || query.outcomes.iter().any(|v| !components.iter().any(|component| component.contains(v)))
        || query
            .treatments
            .iter()
            .any(|v| !components.iter().any(|component| component.contains(v)))
        || components.iter().any(|component| {
            !query.outcomes.iter().any(|v| component.contains(v))
                || !query.treatments.iter().any(|v| component.contains(v))
        })
    {
        return None;
    }
    let [first, second] = components.try_into().ok()?;
    Some([first, second])
}

/// Whether the shared static graph is one connected component under directed,
/// bidirected, and parent adjacency. The disconnected case is decided
/// separately, so the connected route only fires on a single component.
fn graph_is_connected(graph: &antecedent_graph::Admg) -> bool {
    if graph.node_count() == 0
        || graph.nodes().iter().any(|node| !matches!(node, NodeRef::Static(_)))
    {
        return false;
    }
    graph.weak_components().len() == 1
}

/// The `do(treatments)`-mutilated ADMG on the same dense node layout: every
/// directed edge into a treatment and every bidirected edge incident to a
/// treatment is dropped, so the treatments are exogenous roots. m-separation on
/// this graph is m-separation in the intervened law `P*_x(v)`.
fn mutilated_admg(
    graph: &antecedent_graph::Admg,
    treatments: &BitSet,
) -> Result<antecedent_graph::Admg, IdentificationError> {
    let build = || IdentificationError::invalid_input("z_transport.mutilated_graph");
    graph
        .copy_filtered(
            antecedent_graph::CopyLabels::StaticIndex { extra: 0 },
            |_from, to| !treatments.contains(to),
            // do(X) severs latent confounding of X.
            |a, b| !treatments.contains(a) && !treatments.contains(b),
        )
        .map_err(|_| build())
}

/// The `do(treatments)`-mutilated form of a static `graph`: directed edges into
/// a treatment and bidirected edges incident to one are dropped, so the
/// treatments are exogenous roots.
///
/// # Errors
/// A treatment that is not a static node of `graph`.
pub fn intervention_mutilated_admg(
    graph: &antecedent_graph::Admg,
    treatments: &[VariableId],
) -> Result<antecedent_graph::Admg, IdentificationError> {
    let mut set = BitSet::with_len(graph.node_count());
    for treatment in treatments {
        set.insert(graph.dense_of_static(*treatment).ok_or_else(build_unknown)?);
    }
    mutilated_admg(graph, &set)
}

/// The `treatments` that remain directed ancestors of `outcomes` in the
/// `do(treatments)`-mutilated graph `mutilated` of `graph`, sorted by id: the
/// treatments whose level still changes the outcomes' interventional law.
///
/// # Errors
/// An outcome that is not a static node of `graph`.
pub fn intervention_ancestral_treatments(
    graph: &antecedent_graph::Admg,
    mutilated: &antecedent_graph::Admg,
    treatments: &[VariableId],
    outcomes: &[VariableId],
) -> Result<Vec<VariableId>, IdentificationError> {
    let seeds = outcomes
        .iter()
        .map(|variable| graph.dense_of_static(*variable).ok_or_else(build_unknown))
        .collect::<Result<Vec<_>, _>>()?;
    let mut ancestors = BitSet::with_len(mutilated.node_count());
    mutilated.ancestors_of(
        &seeds,
        &mut ancestors,
        &mut antecedent_graph::GraphWorkspace::default(),
    );
    let mut retained = treatments
        .iter()
        .copied()
        .filter(|treatment| {
            graph.dense_of_static(*treatment).is_some_and(|dense| ancestors.contains(dense))
        })
        .collect::<Vec<_>>();
    retained.sort_unstable_by_key(|variable| variable.raw());
    Ok(retained)
}

/// Partition the queried outcomes into exactly two groups that are m-separated
/// from each other given the treatments in the `do(X)`-mutilated graph, so
/// `P*_x(y)` factorizes as the product of the two groups' marginal effects.
/// `None` when the graph is not all-static, the outcomes stay m-connected (one
/// group, whose single c-factor would need a fabricated cross-source joint), or
/// they split into more than two groups (outside the two-source contract).
fn intervention_separated_outcome_groups(
    graph: &antecedent_graph::Admg,
    query: &TwoSourceZTransportQuery,
) -> Result<Option<[Vec<VariableId>; 2]>, IdentificationError> {
    if graph.nodes().iter().any(|node| !matches!(node, NodeRef::Static(_))) {
        return Ok(None);
    }
    let mut treatments = BitSet::with_len(graph.node_count());
    for treatment in query.treatments.iter().copied() {
        let Some(dense) = graph.dense_of_static(treatment) else { return Ok(None) };
        treatments.insert(dense);
    }
    let mutilated = mutilated_admg(graph, &treatments)?;
    let treatment_ids = treatments.to_dense_ids();
    let outcomes = query.outcomes.to_vec();
    let outcome_dense = outcomes
        .iter()
        .map(|variable| graph.dense_of_static(*variable).ok_or_else(build_unknown))
        .collect::<Result<Vec<_>, _>>()?;
    // Connect two outcomes that are m-connected given the treatments; the groups
    // are the connected components of that relation. Cross-group pairs are all
    // m-separated, so the groups are jointly independent given the intervention.
    let mut ws = antecedent_graph::DSeparationWorkspace::default();
    let mut labels = vec![usize::MAX; outcomes.len()];
    let mut group_count = 0usize;
    for start in 0..outcomes.len() {
        if labels[start] != usize::MAX {
            continue;
        }
        let label = group_count;
        group_count += 1;
        let mut stack = vec![start];
        labels[start] = label;
        while let Some(current) = stack.pop() {
            for other in 0..outcomes.len() {
                if labels[other] != usize::MAX {
                    continue;
                }
                let connected = !mutilated
                    .is_m_separated(
                        outcome_dense[current],
                        outcome_dense[other],
                        &treatment_ids,
                        &mut ws,
                    )
                    .map_err(|_| build_unknown())?;
                if connected {
                    labels[other] = label;
                    stack.push(other);
                }
            }
        }
    }
    if group_count != 2 {
        return Ok(None);
    }
    let mut groups: [Vec<VariableId>; 2] = [Vec::new(), Vec::new()];
    for (index, variable) in outcomes.into_iter().enumerate() {
        groups[labels[index]].push(variable);
    }
    Ok(Some(groups))
}

fn build_unknown() -> IdentificationError {
    IdentificationError::invalid_input("z_transport.unknown_variable")
}

/// Whether `current` fixes a treatment that an earlier group already fixed at a
/// different level. Both factors of a product must describe one intervention:
/// a treatment that drives both groups is set to the same level in each source,
/// or the product is not the law of any single target regime.
#[must_use]
pub fn intervention_levels_conflict(
    earlier: &[Vec<CatalogInterventionAssignment>],
    current: &[CatalogInterventionAssignment],
) -> bool {
    earlier.iter().flatten().any(|previous| {
        current.iter().any(|assignment| {
            assignment.variable == previous.variable
                && !antecedent_core::same_intervention_level(&assignment.value, &previous.value)
        })
    })
}

/// Decide the connected complementary case: one connected graph whose intervened
/// outcomes factorize into two groups, each transported from one distinct
/// source. Each factor binds only to its own source's regimes (the joint-regime
/// rule); a c-factor that would span both sources is never fabricated.
fn connected_complementary_decision(
    graph: &antecedent_graph::Admg,
    query: &TwoSourceZTransportQuery,
    catalogs: [&EvidenceCatalog; 2],
    limits: super::super::SidLimits,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<Option<TwoSourceZTransportDecision>, IdentificationError> {
    if !graph_is_connected(graph) {
        return Ok(None);
    }
    let Some(groups) = intervention_separated_outcome_groups(graph, query)? else {
        return Ok(None);
    };
    let mut treatments = BitSet::with_len(graph.node_count());
    for treatment in query.treatments.iter().copied() {
        let Some(dense) = graph.dense_of_static(treatment) else { return Ok(None) };
        treatments.insert(dense);
    }
    let mutilated = mutilated_admg(graph, &treatments)?;
    // Each group's marginal effect P*_x(y_g) equals P*_{x_g}(y_g), where x_g are
    // the treatments that remain ancestors of the group after intervention; the
    // other treatments do not change the group's law.
    let relevant_treatments = |group: &[VariableId]| {
        intervention_ancestral_treatments(graph, &mutilated, &query.treatments, group)
    };
    for order in [[0usize, 1usize], [1usize, 0usize]] {
        let mut proofs = Vec::with_capacity(2);
        let mut group_assignments: Vec<Vec<CatalogInterventionAssignment>> = Vec::with_capacity(2);
        let mut complete = true;
        for (group_index, source_index) in order.into_iter().enumerate() {
            let outcomes = &groups[group_index];
            let source = &query.sources[source_index];
            let group_treatments = relevant_treatments(outcomes)?;
            let assignment = source
                .experiment_assignment
                .iter()
                .filter(|assignment| group_treatments.contains(&assignment.variable))
                .cloned()
                .collect::<Vec<_>>();
            if group_treatments.is_empty()
                || group_treatments.iter().any(|variable| !source.controllable.contains(variable))
                || !group_treatments.iter().all(|variable| {
                    assignment.iter().any(|assignment| assignment.variable == *variable)
                })
                || intervention_levels_conflict(&group_assignments, &assignment)
            {
                complete = false;
                break;
            }
            group_assignments.push(assignment.clone());
            let component_query = ZTransportQuery {
                outcomes: outcomes.clone().into(),
                treatments: group_treatments.clone().into(),
                controllable: Arc::clone(&source.controllable),
                experiment_assignment: assignment.into(),
                source: Arc::clone(&source.population),
                target: Arc::clone(&query.target),
            };
            let diagram =
                SelectionDiagram::try_new(graph.clone(), Arc::clone(&source.selection_targets))
                    .map_err(|error| IdentificationError::invalid_input(error.to_string()))?;
            let ZTransportDecision::Identified(derivation) = decide_z_transport_with_catalog(
                &diagram,
                &component_query,
                catalogs[source_index],
                limits,
                ctx,
            )?
            else {
                complete = false;
                break;
            };
            proofs.push(TwoSourceZTransportComponent {
                source: Arc::clone(&source.population),
                outcomes: outcomes.clone().into(),
                treatments: group_treatments.into(),
                derivation,
            });
        }
        if complete {
            let [first, second] = proofs.try_into().expect("two connected components");
            return Ok(Some(TwoSourceZTransportDecision::CombinedIdentified {
                components: [first, second],
                factorization: ComponentFactorization::InterventionSeparatedGroups,
            }));
        }
    }
    Ok(None)
}

pub(super) fn certify_line11_obstruction(
    diagram: &SelectionDiagram,
    query: &ZTransportQuery,
    terminal: TrzTerminalFailure,
    meter: super::super::SidMeter<'_>,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<ZTransportDecision, IdentificationError> {
    let obstruction = ZTransportObstruction {
        query: query.clone(),
        graph_signature: super::super::graph_signature(diagram),
        selected_assignment: query
            .experiment_assignment
            .iter()
            .filter_map(|assignment| {
                assignment.value.as_f64().map(|value| (assignment.variable.raw(), value))
            })
            .collect(),
        terminal,
    };
    verify_z_obstruction_metered(diagram, query, &obstruction, meter, ctx)?;
    Ok(ZTransportDecision::ProvenNonTransportable(Box::new(obstruction)))
}
