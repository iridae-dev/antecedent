//! Temporal cross-world contrast construction and evaluation.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    Arc, CausalError, CausalExprArena, CausalQuery, DerivationMeta, DomainRef, EstimandMethod,
    ExprNode, IdentificationResult, IdentifiedEstimand, Intervention, OutcomeExprId, TemporalDag,
    VariableId,
};

/// The first node of a temporal path-specific effect's path, or the refusal when the path
/// names no node (every directed path is the total effect, not a path product) or names a node
/// twice.
pub(in crate::analysis::execute) fn require_path_nodes(
    path: &[VariableId],
) -> Result<VariableId, CausalError> {
    let Some(&first) = path.first() else {
        return Err(CausalError::Unsupported {
            message: "temporal path-specific effect needs at least one path node",
        });
    };
    if path.iter().enumerate().any(|(index, node)| path[..index].contains(node)) {
        return Err(CausalError::Unsupported {
            message: "temporal path-specific effect names a path node twice",
        });
    }
    Ok(first)
}

/// Refuses a multi-node path whose nodes are not a contemporaneous directed chain of `graph` in
/// the order listed. The temporal product reads the nodes as one path from the treatment to the
/// outcome in listed order, so an order the graph contradicts would multiply the wrong edges.
pub(in crate::analysis::execute) fn require_path_chain(
    graph: &TemporalDag,
    path: &[VariableId],
) -> Result<(), CausalError> {
    let dense = |variable: VariableId| {
        (0..graph.node_count()).map(antecedent_graph::DenseNodeId::from_index).find(|&id| {
            graph.temporal_key(id).is_some_and(|key| key.variable == variable && key.offset == 0)
        })
    };
    for pair in path.windows(2) {
        let reaches = match (dense(pair[0]), dense(pair[1])) {
            (Some(from), Some(to)) => graph.reaches(from, to),
            _ => false,
        };
        if !reaches {
            return Err(CausalError::Unsupported {
                message: "temporal path-specific path nodes must be listed in causal order, each contemporaneous node reaching the next in the graph",
            });
        }
    }
    Ok(())
}

/// The product of the structural edge coefficients along a treatment-to-outcome path, from the
/// regressions `fit` returns: the first node on the treatment, then each node's successor (the
/// outcome after the last) on the treatment and every path node up to it.
fn path_edge_product(
    fit: &impl Fn(VariableId, &[VariableId], &[VariableId]) -> Result<(f64, Vec<f64>), CausalError>,
    treatment: VariableId,
    outcome: VariableId,
    path: &[VariableId],
) -> Result<f64, CausalError> {
    let first = require_path_nodes(path)?;
    let (_, through) = fit(first, &[treatment], &[])?;
    let mut product = through[0];
    for index in 0..path.len() {
        let child = path.get(index + 1).copied().unwrap_or(outcome);
        let mut regressors = Vec::with_capacity(index + 2);
        regressors.push(treatment);
        regressors.extend_from_slice(&path[..=index]);
        let (_, slopes) = fit(child, &regressors, &path[..=index])?;
        product *= slopes[index + 1];
    }
    Ok(product)
}

/// The treatment, outcome, mediator and levels of a cross-world query, or the
/// refusal when the query is not a mediated or hard-intervention contrast.
pub(super) fn cross_world_contrast(query: &CausalQuery) -> Result<CrossWorldContrast, CausalError> {
    let (treatment, outcome, mediator, active, control) =
        super::super::temporal_class_point::cross_world_endpoints(query)?;
    Ok(CrossWorldContrast { treatment, outcome, mediator, active, control })
}

/// The linear unfolded cross-world point of the query's contrast, from the
/// fitted slopes of `fit`.
pub(in crate::analysis::execute) fn cross_world_point(
    query: &CausalQuery,
    fit: &impl Fn(VariableId, &[VariableId], &[VariableId]) -> Result<(f64, Vec<f64>), CausalError>,
    contrast: CrossWorldContrast,
) -> Result<f64, CausalError> {
    let CrossWorldContrast { treatment, outcome, mediator, active, control } = contrast;
    let delta = active - control;
    Ok(match query {
        CausalQuery::NestedCounterfactual(_) => {
            let mediator = mediator.ok_or_else(missing_cross_world_mediator)?;
            let (_, slopes) = fit(outcome, &[treatment, mediator], &[mediator])?;
            slopes[0] * delta
        }
        CausalQuery::PathSpecific(path) => {
            let _ = mediator.ok_or_else(missing_cross_world_mediator)?;
            path_edge_product(fit, treatment, outcome, &path.path_nodes)? * delta
        }
        CausalQuery::Counterfactual(_) => {
            let (_, slopes) = fit(outcome, &[treatment], &[])?;
            slopes[0] * delta
        }
        CausalQuery::Distribution(_) => {
            let (intercept, slopes) = fit(outcome, &[treatment], &[])?;
            intercept + slopes[0] * active
        }
        _ => Err(CausalError::Unsupported {
            message: "temporal cross-world route does not estimate this query",
        })?,
    })
}

/// A mediated cross-world contrast whose contrast record carries no mediator.
fn missing_cross_world_mediator() -> CausalError {
    CausalError::Compile { message: "cross-world mediated contrast needs a mediator".into() }
}

/// The variables and levels of one cross-world (mediation) contrast.
#[derive(Clone, Copy)]
pub(in crate::analysis::execute) struct CrossWorldContrast {
    pub(in crate::analysis::execute) treatment: VariableId,
    pub(in crate::analysis::execute) outcome: VariableId,
    pub(in crate::analysis::execute) mediator: Option<VariableId>,
    pub(in crate::analysis::execute) active: f64,
    pub(in crate::analysis::execute) control: f64,
}

pub(in crate::analysis::execute) fn template_scoped_cross_world(
    mut identification: IdentificationResult,
    indexer: &antecedent_core::TemporalIndexer,
    query: CausalQuery,
    contrast: CrossWorldContrast,
) -> IdentificationResult {
    let CrossWorldContrast { treatment, outcome, mediator, active, control } = contrast;
    let map = |id: VariableId| indexer.key_of(id.raw()).map_or(id, |key| key.variable);
    let adjustment: Vec<VariableId> = identification
        .estimands
        .first()
        .map(|estimand| {
            let mut mapped: Vec<VariableId> =
                estimand.adjustment_set.iter().copied().map(map).collect();
            mapped.sort_by_key(|id| id.raw());
            mapped.dedup();
            mapped
        })
        .unwrap_or_default();
    let mut arena = CausalExprArena::new();
    let backdoor_method = identification.estimands.first().map_or_else(
        || Arc::from("temporal.backdoor.unfolded"),
        |estimand| estimand.method.clone(),
    );
    let active_value = antecedent_core::Value::f64(active);
    let control_value = antecedent_core::Value::f64(control);
    // Nested and path points are the direct slope and the mediator-edge product
    // that were fit, under the same adjustment set. A counterfactual and an interventional mean stay the
    // back-door formula of the adjustment set that entered that fit.
    let (method, adjustment, mediators, functional) = match (&query, mediator) {
        (CausalQuery::NestedCounterfactual(_), Some(mediator)) => (
            Arc::from(EstimandMethod::TemporalMediationDirect.as_str()),
            Arc::from(adjustment),
            Arc::from([mediator]),
            direct_slope_functional(&mut arena, treatment, mediator, outcome, active, control),
        ),
        (CausalQuery::PathSpecific(path), Some(_)) => (
            Arc::from(EstimandMethod::PathSpecificNatural.as_str()),
            Arc::from(adjustment),
            Arc::clone(&path.path_nodes),
            path_edge_product_functional(
                &mut arena,
                (treatment, outcome),
                &path.path_nodes,
                (active, control),
            ),
        ),
        (CausalQuery::Distribution(_), _) => {
            let functional = arena.backdoor_mean(treatment, outcome, &adjustment, active_value);
            (backdoor_method, Arc::from(adjustment), Arc::from([]), functional)
        }
        _ => {
            let functional =
                arena.backdoor_ate(treatment, outcome, &adjustment, active_value, control_value);
            (backdoor_method, Arc::from(adjustment), Arc::from([]), functional)
        }
    };
    identification.query = query;
    identification.arena = arena;
    identification.estimands = vec![IdentifiedEstimand::new(
        method,
        adjustment,
        Arc::from([]),
        mediators,
        functional,
        None,
    )];
    identification.hedge = None;
    identification
}

/// Linear natural direct effect: the treatment slope in the outcome regression
/// that holds the contemporaneous mediator, times the level gap.
fn direct_slope_functional(
    arena: &mut CausalExprArena,
    treatment: VariableId,
    mediator: VariableId,
    outcome: VariableId,
    active: f64,
    control: f64,
) -> antecedent_expr::ExprId {
    let left = held_mediator_mean(arena, treatment, mediator, outcome, active);
    let right = held_mediator_mean(arena, treatment, mediator, outcome, control);
    let contrast = arena.intern(ExprNode::Contrast {
        left,
        right,
        op: antecedent_expr::ContrastOp::Difference,
    });
    arena.set_derivation(
        contrast,
        DerivationMeta::rule(
            "temporal.cross_world.direct_slope",
            Some(Arc::from(
                "outcome-regression treatment slope with the contemporaneous mediator held, times the level gap",
            )),
        ),
    );
    contrast
}

fn held_mediator_mean(
    arena: &mut CausalExprArena,
    treatment: VariableId,
    mediator: VariableId,
    outcome: VariableId,
    level: f64,
) -> antecedent_expr::ExprId {
    let variables = arena.intern_var_set([outcome]);
    let conditioned_on = arena.intern_var_set([treatment, mediator]);
    let intervention =
        arena.intern_intervention_assignments([antecedent_expr::InterventionAssignment::concrete(
            treatment,
            antecedent_core::Value::f64(level),
        )]);
    let distribution = arena.intern_distribution(
        variables,
        conditioned_on,
        intervention,
        DomainRef::Interventional,
    );
    arena.intern(ExprNode::Expectation { function: OutcomeExprId::identity(outcome), distribution })
}

/// Product of the treatment-to-first-node edge over the level gap and, along the path, each
/// node's partial slope on its successor (the outcome after the last node) holding the
/// treatment and the earlier path nodes.
fn path_edge_product_functional(
    arena: &mut CausalExprArena,
    (treatment, outcome): (VariableId, VariableId),
    path: &[VariableId],
    (active, control): (f64, f64),
) -> antecedent_expr::ExprId {
    let mut factors = Vec::with_capacity(path.len() + 1);
    factors.push(arena.backdoor_ate(
        treatment,
        path[0],
        &[],
        antecedent_core::Value::f64(active),
        antecedent_core::Value::f64(control),
    ));
    for index in 0..path.len() {
        let child = path.get(index + 1).copied().unwrap_or(outcome);
        let held = &path[..=index];
        let high = held_path_mean(arena, (treatment, control), held, (child, 1.0));
        let low = held_path_mean(arena, (treatment, control), held, (child, 0.0));
        factors.push(arena.intern(ExprNode::Contrast {
            left: high,
            right: low,
            op: antecedent_expr::ContrastOp::Difference,
        }));
    }
    let factors = arena.intern_list(factors);
    let product = arena.intern(ExprNode::Product(factors));
    arena.set_derivation(
        product,
        DerivationMeta::rule(
            "temporal.cross_world.mediator_edge_product",
            Some(Arc::from(if path.len() == 1 {
                "product of the treatment-to-mediator slope and the mediator-to-outcome slope, times the level gap"
            } else {
                "product of the treatment-to-first-node slope and each path node's slope on its successor along the path, times the level gap"
            })),
        ),
    );
    product
}

/// `E[child | do(treatment = treatment_level, earlier path nodes = 0, last held node = level)]`.
fn held_path_mean(
    arena: &mut CausalExprArena,
    (treatment, treatment_level): (VariableId, f64),
    held: &[VariableId],
    (child, level): (VariableId, f64),
) -> antecedent_expr::ExprId {
    let variables = arena.intern_var_set([child]);
    let conditioned_on =
        arena.intern_var_set(std::iter::once(treatment).chain(held.iter().copied()));
    let last = held.len().saturating_sub(1);
    let assignments = std::iter::once(antecedent_expr::InterventionAssignment::concrete(
        treatment,
        antecedent_core::Value::f64(treatment_level),
    ))
    .chain(held.iter().enumerate().map(|(index, &node)| {
        antecedent_expr::InterventionAssignment::concrete(
            node,
            antecedent_core::Value::f64(if index == last { level } else { 0.0 }),
        )
    }));
    let intervention = arena.intern_intervention_assignments(assignments);
    let distribution = arena.intern_distribution(
        variables,
        conditioned_on,
        intervention,
        DomainRef::Interventional,
    );
    arena.intern(ExprNode::Expectation { function: OutcomeExprId::identity(child), distribution })
}

fn hard_level(intervention: &Intervention) -> Result<f64, CausalError> {
    match intervention {
        Intervention::Set { value: antecedent_core::Value::Float64(level), .. }
            if level.is_finite() =>
        {
            Ok(*level)
        }
        _ => Err(CausalError::Unsupported {
            message: "temporal cross-world interventions are finite hard sets",
        }),
    }
}

pub(in crate::analysis::execute) fn temporal_cross_world_hard_level(
    intervention: &Intervention,
) -> Result<f64, CausalError> {
    hard_level(intervention)
}
