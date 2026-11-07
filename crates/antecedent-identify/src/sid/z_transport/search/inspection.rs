//! Inspect z-transport expressions and construct registered surrogate formulas.

use super::{
    Arc, CausalExprArena, DistributionAvailability, DomainRef, EvidenceCatalog, ExprId, ExprNode,
    IdentificationError, NodeRef, SelectionDiagram, VariableId, ZFactorObligation, ZProofOperation,
    ZTransportProofInspection, ZTransportQuery, index_u32,
};

use super::binding::regime_matches_world;

pub(in crate::sid::z_transport) fn inspect_z_expression(
    id: ExprId,
    arena: &CausalExprArena,
    catalog: &EvidenceCatalog,
    seen: &mut std::collections::BTreeSet<u32>,
    result: &mut ZTransportProofInspection,
) {
    if !seen.insert(id.raw()) {
        return;
    }
    let node = arena.node(id);
    let (operation, children): (&str, Vec<ExprId>) = match node {
        ExprNode::Distribution { .. } => ("distribution", Vec::new()),
        ExprNode::Product(list) => ("product", arena.list(*list).to_vec()),
        ExprNode::SumOut { expr, .. } => ("sum_out", vec![*expr]),
        ExprNode::Ratio { numerator, denominator } => ("ratio", vec![*numerator, *denominator]),
        ExprNode::Kernel { body, .. } => ("kernel", vec![*body]),
        ExprNode::IntegralOut { expr, .. } => ("integral_out", vec![*expr]),
        ExprNode::Expectation { distribution, .. } => ("expectation", vec![*distribution]),
        ExprNode::Contrast { left, right, .. } => ("contrast", vec![*left, *right]),
    };
    for child in &children {
        inspect_z_expression(*child, arena, catalog, seen, result);
    }
    result.operations.push(ZProofOperation {
        id: id.raw(),
        operation: operation.to_owned(),
        children: children.iter().map(|child| child.raw()).collect(),
    });
    let ExprNode::Distribution { variables, conditioned_on, intervention, population, .. } = node
    else {
        return;
    };
    let name = arena.population(*population);
    let vars = arena.var_set(*variables);
    let conditions = arena.var_set(*conditioned_on);
    let assignments = arena.intervention_assignments(*intervention);
    let mut exact_regime = false;
    let mut missing_joint = false;
    let mut missing_margin = false;
    let mut missing_binding = false;
    let mut supplied_by = Vec::new();
    for regime in catalog.regimes.iter().filter(|regime| {
        regime.population.as_ref() == name
            && regime.supplies_population_law()
            && regime_matches_world(regime, assignments)
    }) {
        exact_regime = true;
        if regime.distribution != DistributionAvailability::Joint
            || !regime.conditioned_on.is_empty()
        {
            missing_joint = true;
        } else if !vars.iter().chain(conditions).all(|v| regime.measured.contains(v)) {
            missing_margin = true;
        } else if !catalog.bindings.iter().any(|binding| binding.regime == regime.id) {
            missing_binding = true;
        } else {
            supplied_by.push(regime.id.raw());
        }
    }
    let failure = if !supplied_by.is_empty() {
        None
    } else if !exact_regime {
        Some("missing_matching_population_regime_or_intervention_values")
    } else if missing_joint {
        Some("joint_law_required")
    } else if missing_margin {
        Some("measured_joint_margin_insufficient")
    } else if missing_binding {
        Some("provider_binding_missing")
    } else {
        Some("factor_binding_failed")
    };
    result.factors.push(ZFactorObligation {
        leaf: id.raw(),
        population: name.to_owned(),
        variables: vars.iter().map(|v| v.raw()).collect(),
        conditioned_on: conditions.iter().map(|v| v.raw()).collect(),
        intervention: assignments.iter().map(|a| (a.variable.raw(), a.value.as_f64())).collect(),
        supplied_by,
        failure: failure.map(str::to_owned),
    });
}

/// The registered surrogate formula `Σ_w P_s(y | w, x, do(z)) P_s(w | do(z))`.
pub(in crate::sid::z_transport) fn surrogate_formula(
    query: &ZTransportQuery,
    confounder: VariableId,
) -> Result<(CausalExprArena, ExprId), IdentificationError> {
    let mut arena = CausalExprArena::new();
    let y_set = arena.intern_var_set([query.outcomes[0]]);
    let w_set = arena.intern_var_set([confounder]);
    let wx_set = arena.intern_var_set([confounder, query.treatments[0]]);
    let z_do = crate::intervention_support::intern_assignments(
        &mut arena,
        query.experiment_assignment.iter().map(|a| {
            antecedent_expr::InterventionAssignment::concrete(a.variable, a.value.clone())
        }),
    )?;
    let source = arena.intern_population(Arc::clone(&query.source));
    let conditional = arena.intern(ExprNode::Distribution {
        variables: y_set,
        conditioned_on: wx_set,
        intervention: z_do,
        domain: DomainRef::Interventional,
        population: source,
        regime: None,
    });
    let empty = arena.empty_var_set();
    let marginal = arena.intern(ExprNode::Distribution {
        variables: w_set,
        conditioned_on: empty,
        intervention: z_do,
        domain: DomainRef::Interventional,
        population: source,
        regime: None,
    });
    let factors = arena.intern_list([conditional, marginal]);
    let product = arena.intern(ExprNode::Product(factors));
    let root = arena.intern(ExprNode::SumOut { variables: w_set, expr: product });
    Ok((arena, root))
}

pub(in crate::sid::z_transport) fn matches_registered_surrogate_graph(
    diagram: &SelectionDiagram,
    w: VariableId,
    z: VariableId,
    x: VariableId,
    y: VariableId,
) -> bool {
    let graph = diagram.causal_graph();
    let dense = |variable: VariableId| {
        graph.nodes().iter().position(|node| *node == NodeRef::Static(variable))
    };
    let Some(wi) = dense(w) else { return false };
    let Some(zi) = dense(z) else { return false };
    let Some(xi) = dense(x) else { return false };
    let Some(yi) = dense(y) else { return false };
    let directed = graph
        .nodes()
        .iter()
        .enumerate()
        .flat_map(|(from, _)| {
            graph
                .children(antecedent_graph::DenseNodeId::from_raw(index_u32(from)))
                .iter()
                .map(move |to| (from, to.as_usize()))
        })
        .collect::<std::collections::BTreeSet<_>>();
    let expected_directed = [(wi, zi), (zi, xi), (xi, yi), (wi, yi)]
        .into_iter()
        .collect::<std::collections::BTreeSet<_>>();
    let bidirected = graph
        .nodes()
        .iter()
        .enumerate()
        .flat_map(|(a, _)| {
            graph
                .bidirected_neighbors(antecedent_graph::DenseNodeId::from_raw(index_u32(a)))
                .iter()
                .filter(move |b| a < b.as_usize())
                .map(move |b| (a, b.as_usize()))
        })
        .collect::<std::collections::BTreeSet<_>>();
    let expected_bidirected =
        [(wi.min(yi), wi.max(yi)), (zi.min(yi), zi.max(yi)), (zi.min(xi), zi.max(xi))]
            .into_iter()
            .collect::<std::collections::BTreeSet<_>>();
    directed == expected_directed && bidirected == expected_bidirected
}

/// Set equality of two coordinate lists: repeats and order do not matter.
pub(in crate::sid::z_transport) fn same_variable_set(
    left: &[VariableId],
    right: &[VariableId],
) -> bool {
    left.iter().collect::<std::collections::BTreeSet<_>>()
        == right.iter().collect::<std::collections::BTreeSet<_>>()
}
