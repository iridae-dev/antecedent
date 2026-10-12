//! Recursive z-transport search and catalog binding.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::*;
use super::*;

/// Run the `TRz` search, reporting an exhausted engine budget as the
/// z-transport budget. This is the one place that remap happens.
pub(super) fn z_search(
    diagram: &SelectionDiagram,
    query: &ZTransportQuery,
    meter: super::super::SidMeter<'_>,
    ctx: &antecedent_core::ExecutionContext,
    receipt: &mut Option<ZTransportLimitsReceipt>,
) -> Result<TrzSearchResult, IdentificationError> {
    search_trz_detailed(diagram, query, meter, ctx, receipt).map_err(|error| match error {
        IdentificationError::Budget { budget: IdentificationBudget::Steps } => {
            IdentificationError::budget(IdentificationBudget::ZTransport)
        }
        other => other,
    })
}

/// Bind the checked surrogate formula to a sufficient source joint margin.
///
/// When several available regimes supply the same factor (or the same world of
/// a factor whose exchanged coordinate is bound at evaluation), every formula
/// kind binds by one deterministic rule: prefer a family regime, one declaring
/// no intervention levels, otherwise the lowest regime id. Duplicate evidence
/// is never refused, and the choice does not depend on catalog order.
///
/// # Errors
/// Invalid catalog or a missing joint margin covering the formula variables.
#[allow(
    clippy::too_many_lines,
    reason = "one linear identification pipeline whose stages share the prepared graph and workspace"
)]
pub fn bind_z_transport_catalog(
    diagram: &SelectionDiagram,
    query: &ZTransportQuery,
    derivation: &ZTransportDerivation,
    catalog: &EvidenceCatalog,
) -> Result<BoundZTransportFunctional, IdentificationError> {
    derivation.check_inputs(diagram, query)?;
    catalog.validate().map_err(|error| {
        IdentificationError::invalid_catalog(format!("z_transport.invalid_catalog: {error}"))
    })?;
    if derivation.kind == ZFormulaKind::Recursive {
        let mut arena = derivation.arena.clone();
        let mut memo = std::collections::HashMap::new();
        let mut cited = Vec::new();
        let root = bind_recursive_expression(
            derivation.root,
            &mut arena,
            catalog,
            &query.treatments,
            &mut memo,
            &mut cited,
        )?;
        if cited.is_empty() {
            return Err(IdentificationError::InvariantViolated {
                message: "z_transport.recursive_formula_has_no_factor",
            });
        }
        return Ok(BoundZTransportFunctional {
            derivation: derivation.clone(),
            arena,
            root,
            catalog: catalog.clone(),
            cited: cited_regimes(cited),
        });
    }
    if derivation.kind == ZFormulaKind::DirectJoint {
        let assignment_variables =
            query.experiment_assignment.iter().map(|a| a.variable).collect::<Vec<_>>();
        let selected = preferred_regime(catalog.regimes.iter().filter(|regime| {
            regime.population.as_ref() == query.source.as_ref()
                && regime.supplies_population_law()
                && regime.kind == RegimeKind::Experimental
                && same_variable_set(&regime.interventions, &assignment_variables)
                && regime.intervention_values.len() == query.experiment_assignment.len()
                && query.experiment_assignment.iter().all(|expected| {
                    regime.intervention_values.iter().any(|actual| {
                        actual.variable == expected.variable
                            && antecedent_core::same_intervention_level(
                                &actual.value,
                                &expected.value,
                            )
                    })
                })
                && query.outcomes.iter().all(|y| regime.measured.contains(y))
                && regime.conditioned_on.is_empty()
                && regime.distribution == DistributionAvailability::Joint
                && catalog.bindings.iter().any(|binding| binding.regime == regime.id)
        }))
        .ok_or_else(|| {
            IdentificationError::missing_evidence(
                "z_transport.missing_evidence",
                format!(
                    "{} joint law under do({:?}) measuring {:?}",
                    query.source, query.experiment_assignment, query.outcomes
                ),
            )
        })?;
        let mut arena = derivation.arena.clone();
        let ExprNode::Distribution {
            variables,
            conditioned_on,
            intervention,
            domain,
            population,
            ..
        } = arena.node(derivation.root).clone()
        else {
            return Err(IdentificationError::InvariantViolated {
                message: "z_transport.invalid_formula_root",
            });
        };
        let root = arena.intern(ExprNode::Distribution {
            variables,
            conditioned_on,
            intervention,
            domain,
            population,
            regime: Some(selected.id),
        });
        return Ok(BoundZTransportFunctional {
            derivation: derivation.clone(),
            arena,
            root,
            catalog: catalog.clone(),
            cited: cited_regimes(vec![selected.id]),
        });
    }
    let assignments = &query.experiment_assignment;
    let mut intervention_variables = assignments.iter().map(|a| a.variable).collect::<Vec<_>>();
    intervention_variables.sort_unstable();
    // Both source leaves can be obtained by marginalizing a joint law on
    // {Y, W, X}. Requiring every observed coordinate (or the complete
    // experimental family) here would turn a sufficient positive certificate
    // into a spurious missing-evidence result.
    let Some(confounder) = derivation.confounder else {
        return Err(IdentificationError::InvariantViolated {
            message: "z_transport.surrogate_formula_names_no_confounder",
        });
    };
    let required_margin = [query.outcomes[0], confounder, query.treatments[0]];
    let selected = preferred_regime(catalog.regimes.iter().filter(|regime| {
        regime.population.as_ref() == query.source.as_ref()
            && regime.supplies_population_law()
            && regime.kind == RegimeKind::Experimental
            && same_variable_set(&regime.interventions, &intervention_variables)
            && regime.intervention_values.len() == assignments.len()
            && assignments.iter().all(|expected| {
                regime.intervention_values.iter().any(|actual| {
                    actual.variable == expected.variable
                        && antecedent_core::same_intervention_level(&actual.value, &expected.value)
                })
            })
            && required_margin.iter().all(|variable| regime.measured.contains(variable))
            && regime.conditioned_on.is_empty()
            && regime.distribution == DistributionAvailability::Joint
            && catalog.bindings.iter().any(|binding| binding.regime == regime.id)
    }))
    .ok_or_else(|| {
        IdentificationError::missing_evidence(
            "z_transport.missing_evidence",
            format!(
                "{} joint law under do({:?}) measuring {:?}",
                query.source, query.experiment_assignment, required_margin
            ),
        )
    })?;

    let mut arena = derivation.arena.clone();
    let ExprNode::SumOut { variables, expr } = derivation.arena.node(derivation.root).clone()
    else {
        return Err(IdentificationError::InvariantViolated {
            message: "z_transport.invalid_formula_root",
        });
    };
    let ExprNode::Product(factors) = derivation.arena.node(expr).clone() else {
        return Err(IdentificationError::InvariantViolated {
            message: "z_transport.invalid_formula_product",
        });
    };
    let mut bound_factors = Vec::new();
    for factor in derivation.arena.list(factors) {
        let ExprNode::Distribution {
            variables,
            conditioned_on,
            intervention,
            domain,
            population,
            ..
        } = derivation.arena.node(*factor).clone()
        else {
            return Err(IdentificationError::InvariantViolated {
                message: "z_transport.invalid_formula_leaf",
            });
        };
        bound_factors.push(arena.intern(ExprNode::Distribution {
            variables,
            conditioned_on,
            intervention,
            domain,
            population,
            regime: Some(selected.id),
        }));
    }
    let bound_list = arena.intern_list(bound_factors);
    let bound_product = arena.intern(ExprNode::Product(bound_list));
    let root = arena.intern(ExprNode::SumOut { variables, expr: bound_product });
    Ok(BoundZTransportFunctional {
        derivation: derivation.clone(),
        arena,
        root,
        catalog: catalog.clone(),
        cited: cited_regimes(vec![selected.id]),
    })
}

pub(crate) fn cited_regimes(
    mut cited: Vec<antecedent_core::RegimeId>,
) -> Arc<[antecedent_core::RegimeId]> {
    cited.sort_unstable_by_key(|regime| regime.raw());
    cited.dedup();
    cited.into()
}

/// The one rule every binding path uses when several available regimes supply
/// the same law: prefer a family regime (one declaring no intervention levels,
/// so it supplies every world at once), otherwise the lowest regime id.
/// Supplying the same law more than once is never refused as ambiguous; the
/// choice is deterministic and independent of catalog order.
fn preferred_regime<'a>(
    candidates: impl IntoIterator<Item = &'a antecedent_core::EvidenceRegime>,
) -> Option<&'a antecedent_core::EvidenceRegime> {
    candidates
        .into_iter()
        .min_by_key(|regime| (!regime.intervention_values.is_empty(), regime.id.raw()))
}

/// How one bound factor selects its law at evaluation.
enum LeafBinding {
    /// One regime supplies every world the factor asks for.
    Single(antecedent_core::RegimeId),
    /// One regime per world of the factor's symbolic coordinates; the concrete
    /// world selects the law at evaluation, so the leaf carries no regime.
    PerWorld(Vec<antecedent_core::RegimeId>),
}

impl LeafBinding {
    fn regimes(&self) -> Vec<antecedent_core::RegimeId> {
        match self {
            Self::Single(regime) => vec![*regime],
            Self::PerWorld(regimes) => regimes.clone(),
        }
    }
}

/// Whether `regime` is an available experiment on exactly the factor's
/// intervention set whose concrete levels agree with the factor's concrete
/// coordinates. A regime declaring no levels stands for the whole family and
/// agrees with every concrete coordinate; a symbolic coordinate agrees with
/// every level.
fn regime_matches_world(
    regime: &antecedent_core::EvidenceRegime,
    interventions: &[antecedent_expr::InterventionAssignment],
) -> bool {
    let intervention_vars = interventions.iter().map(|a| a.variable).collect::<Vec<_>>();
    regime.kind
        == if interventions.is_empty() {
            RegimeKind::Observational
        } else {
            RegimeKind::Experimental
        }
        && same_variable_set(&regime.interventions, &intervention_vars)
        && (regime.intervention_values.is_empty()
            || (regime.intervention_values.len() == interventions.len()
                && interventions.iter().all(|expected| {
                    expected.is_symbolic()
                        || regime.intervention_values.iter().any(|actual| {
                            actual.variable == expected.variable
                                && antecedent_core::same_intervention_level(
                                    &actual.value,
                                    &expected.value,
                                )
                        })
                })))
}

/// Whether `regime` supplies the joint margin a factor reads.
fn regime_supplies_margin(
    catalog: &EvidenceCatalog,
    regime: &antecedent_core::EvidenceRegime,
    needed: &[VariableId],
) -> bool {
    regime.distribution == DistributionAvailability::Joint
        && regime.conditioned_on.is_empty()
        && needed.iter().all(|v| regime.measured.contains(v))
        && catalog.bindings.iter().any(|binding| binding.regime == regime.id)
}

/// Declared finite levels of `variable` in `population`, as the numeric codes a
/// per-world regime names them by.
fn declared_levels(
    catalog: &EvidenceCatalog,
    population: &str,
    variable: VariableId,
) -> Option<Vec<f64>> {
    let coordinate = catalog
        .environments
        .iter()
        .find(|environment| environment.identity.as_ref() == population)?
        .variables
        .iter()
        .find(|coordinate| coordinate.variable == variable)?;
    match coordinate.domain {
        VariableDomain::Binary => Some(vec![0.0, 1.0]),
        VariableDomain::Categorical { cardinality } => {
            Some((0..cardinality).map(f64::from).collect())
        }
        _ => None,
    }
}

/// Bind one factor to the available catalog regimes that supply it.
///
/// A symbolic coordinate the evaluation request binds (a queried treatment)
/// cites every level the catalog supplies, and a request at an unsupplied level
/// is refused at evaluation. A symbolic coordinate an enclosing summation binds
/// needs a regime for every declared level, since every summand is evaluated.
fn bind_leaf(
    catalog: &EvidenceCatalog,
    population: &str,
    interventions: &[antecedent_expr::InterventionAssignment],
    needed: &[VariableId],
    request_bound: &[VariableId],
) -> Result<LeafBinding, IdentificationError> {
    let missing = || {
        IdentificationError::missing_evidence(
            "z_transport.missing_evidence",
            format!("{population} joint factor under {interventions:?}"),
        )
    };
    let candidates = catalog
        .regimes
        .iter()
        .filter(|regime| {
            regime.population.as_ref() == population
                && regime.supplies_population_law()
                && regime_matches_world(regime, interventions)
                && regime_supplies_margin(catalog, regime, needed)
        })
        .collect::<Vec<_>>();
    let symbolic = interventions.iter().filter(|a| a.is_symbolic()).collect::<Vec<_>>();
    let preferred = preferred_regime(candidates.iter().copied());
    if symbolic.is_empty() {
        // A concrete factor names one law.
        return preferred.map(|regime| LeafBinding::Single(regime.id)).ok_or_else(missing);
    }
    // A family regime (no declared levels) supplies every world at once.
    if let Some(family) = preferred.filter(|regime| regime.intervention_values.is_empty()) {
        return Ok(LeafBinding::Single(family.id));
    }
    // Otherwise every world of the symbolic coordinates needs its own regime.
    let mut worlds = vec![Vec::new()];
    for assignment in &symbolic {
        let levels = if request_bound.contains(&assignment.variable) {
            let mut supplied = candidates
                .iter()
                .flat_map(|regime| regime.intervention_values.iter())
                .filter(|actual| actual.variable == assignment.variable)
                .filter_map(|actual| actual.value.as_f64())
                .collect::<Vec<_>>();
            supplied.sort_by(f64::total_cmp);
            supplied.dedup();
            if supplied.is_empty() {
                return Err(missing());
            }
            supplied
        } else {
            declared_levels(catalog, population, assignment.variable).ok_or_else(|| {
                IdentificationError::missing_evidence(
                    "z_transport.missing_evidence",
                    format!("{population} declares no finite domain for {:?}", assignment.variable),
                )
            })?
        };
        worlds = worlds
            .into_iter()
            .flat_map(|prefix| {
                levels.iter().map(move |level| {
                    let mut world = prefix.clone();
                    world.push(*level);
                    world
                })
            })
            .collect();
    }
    let mut regimes = Vec::with_capacity(worlds.len());
    for world in worlds {
        let supplying = candidates.iter().copied().filter(|regime| {
            symbolic.iter().zip(&world).all(|(assignment, level)| {
                regime.intervention_values.iter().any(|actual| {
                    actual.variable == assignment.variable && actual.value.as_f64() == Some(*level)
                })
            })
        });
        let Some(regime) = preferred_regime(supplying) else {
            return Err(IdentificationError::missing_evidence(
                "z_transport.missing_evidence",
                format!("{population} joint factor under {interventions:?} at world {world:?}"),
            ));
        };
        regimes.push(regime.id);
    }
    Ok(LeafBinding::PerWorld(regimes))
}

pub(crate) fn bind_recursive_expression(
    id: ExprId,
    arena: &mut CausalExprArena,
    catalog: &EvidenceCatalog,
    treatments: &[VariableId],
    memo: &mut std::collections::HashMap<ExprId, ExprId>,
    cited: &mut Vec<antecedent_core::RegimeId>,
) -> Result<ExprId, IdentificationError> {
    if let Some(hit) = memo.get(&id) {
        return Ok(*hit);
    }
    let node = arena.node(id).clone();
    let bound = match node {
        ExprNode::Distribution {
            variables,
            conditioned_on,
            intervention,
            domain,
            population,
            ..
        } => {
            let needed = arena
                .var_set(variables)
                .iter()
                .chain(arena.var_set(conditioned_on))
                .copied()
                .collect::<Vec<_>>();
            let binding = bind_leaf(
                catalog,
                arena.population(population),
                arena.intervention_assignments(intervention),
                &needed,
                treatments,
            )?;
            let regime = match &binding {
                LeafBinding::Single(regime) => Some(*regime),
                LeafBinding::PerWorld(_) => None,
            };
            cited.extend(binding.regimes());
            arena.intern(ExprNode::Distribution {
                variables,
                conditioned_on,
                intervention,
                domain,
                population,
                regime,
            })
        }
        ExprNode::Product(list) => {
            let children = arena.list(list).to_vec();
            let bound_children = children
                .into_iter()
                .map(|child| {
                    bind_recursive_expression(child, arena, catalog, treatments, memo, cited)
                })
                .collect::<Result<Vec<_>, _>>()?;
            let list = arena.intern_list(bound_children);
            arena.intern(ExprNode::Product(list))
        }
        ExprNode::SumOut { variables, expr } => {
            let expr = bind_recursive_expression(expr, arena, catalog, treatments, memo, cited)?;
            arena.intern(ExprNode::SumOut { variables, expr })
        }
        ExprNode::Ratio { numerator, denominator } => {
            let numerator =
                bind_recursive_expression(numerator, arena, catalog, treatments, memo, cited)?;
            let denominator =
                bind_recursive_expression(denominator, arena, catalog, treatments, memo, cited)?;
            arena.intern(ExprNode::Ratio { numerator, denominator })
        }
        _ => {
            return Err(IdentificationError::InvariantViolated {
                message: "z_transport.unsupported_recursive_node",
            });
        }
    };
    memo.insert(id, bound);
    Ok(bound)
}

pub(super) fn inspect_z_expression(
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
pub(super) fn surrogate_formula(
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

pub(super) fn matches_registered_surrogate_graph(
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
pub(super) fn same_variable_set(left: &[VariableId], right: &[VariableId]) -> bool {
    left.iter().collect::<std::collections::BTreeSet<_>>()
        == right.iter().collect::<std::collections::BTreeSet<_>>()
}

// Bareinboim–Pearl (AAAI 2013, Figure 4) rules 1–4 map to the marginal,
// ancestor, enlargement, and district branches below; rules 5–8 map to the
// C-component, factor, and subdistrict branches. Rule 10 exchanges the active
// controllables in X and recurses on the reduced graph; rule 11 is recorded
// only when that exchange's separation premise fails or Z∩X is empty.
#[cfg(test)]
pub(super) fn search_trz(
    diagram: &SelectionDiagram,
    query: &ZTransportQuery,
    limits: super::super::SidLimits,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<Option<(CausalExprArena, ExprId, Vec<String>)>, IdentificationError> {
    Ok(search_trz_detailed(diagram, query, super::super::SidMeter::Limits(limits), ctx, &mut None)?
        .identified)
}

/// One source a `TRz` line-10 exchange may consult.
pub(crate) struct TrzDomain<'q> {
    pub(crate) population: &'q Arc<str>,
    pub(crate) controllable: &'q [VariableId],
    pub(crate) experiment_assignment: &'q [CatalogInterventionAssignment],
    pub(crate) selection_targets: &'q [VariableId],
}

/// Shared inputs of one recursive search: the queried treatments, the target,
/// and the sources line 10 may exchange into (one for `TRz`, several for `TR^mz`).
pub(crate) struct TrzCall<'q> {
    pub(crate) outcomes: &'q Arc<[VariableId]>,
    pub(crate) treatments: &'q Arc<[VariableId]>,
    pub(crate) target: &'q Arc<str>,
    /// A source population distinct from the target. The engine's classical
    /// query needs one; this recursion never builds a kernel from it, since every
    /// exchanged kernel carries its own source's population.
    pub(crate) anchor: &'q Arc<str>,
    pub(crate) domains: Vec<TrzDomain<'q>>,
}

impl<'q> TrzCall<'q> {
    fn single(diagram: &'q SelectionDiagram, query: &'q ZTransportQuery) -> Self {
        Self {
            outcomes: &query.outcomes,
            treatments: &query.treatments,
            target: &query.target,
            anchor: &query.source,
            domains: vec![TrzDomain {
                population: &query.source,
                controllable: &query.controllable,
                experiment_assignment: &query.experiment_assignment,
                selection_targets: diagram.selection_targets(),
            }],
        }
    }
}

/// One source's line-11 premises at a multi-source terminal.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct TrzSourceTerminal {
    pub(crate) population: String,
    pub(crate) remaining_controllable: Vec<u32>,
    pub(crate) candidate_active: Vec<u32>,
    pub(crate) selection_separated: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct TrzTerminalFailure {
    pub(crate) outcomes: Vec<u32>,
    pub(crate) treatments: Vec<u32>,
    pub(crate) vertices: Vec<u32>,
    pub(crate) c0: Vec<u32>,
    pub(super) remaining_controllable: Vec<u32>,
    pub(super) candidate_active: Vec<u32>,
    pub(crate) active_interventions: Vec<(u32, Option<f64>)>,
    pub(super) selection_separated: bool,
    pub(crate) rules: Vec<String>,
    /// Per-source premises when several sources were searched; empty for `TRz`,
    /// whose single source uses the fields above.
    pub(crate) sources: Vec<TrzSourceTerminal>,
}

#[derive(Debug)]
pub(crate) struct TrzSearchResult {
    pub(crate) identified: Option<(CausalExprArena, ExprId, Vec<String>)>,
    pub(crate) terminal_failure: Option<TrzTerminalFailure>,
    pub(crate) unassigned: Option<VariableId>,
    /// Every recursive rule the search applied, in order, whatever the outcome.
    pub(crate) explored_rules: Vec<String>,
    /// Recursive subproblems the search charged.
    pub(crate) steps_explored: usize,
    /// Deepest recursion level the search reached.
    pub(crate) depth_reached: usize,
    /// A multi-source line-11 state needed a joint over two sources'
    /// interventions, which the search refused to fabricate.
    pub(crate) cross_source_joint: bool,
}

fn search_trz_detailed(
    diagram: &SelectionDiagram,
    query: &ZTransportQuery,
    meter: super::super::SidMeter<'_>,
    ctx: &antecedent_core::ExecutionContext,
    receipt: &mut Option<ZTransportLimitsReceipt>,
) -> Result<TrzSearchResult, IdentificationError> {
    search_trz_call(diagram, &TrzCall::single(diagram, query), meter, ctx, receipt)
}

/// Run the recursive search for `call` on the shared causal graph of `diagram`,
/// charging `meter`. Selection targets are read per source from `call`, never
/// from `diagram`.
pub(crate) fn search_trz_call(
    diagram: &SelectionDiagram,
    call: &TrzCall<'_>,
    meter: super::super::SidMeter<'_>,
    ctx: &antecedent_core::ExecutionContext,
    receipt: &mut Option<ZTransportLimitsReceipt>,
) -> Result<TrzSearchResult, IdentificationError> {
    let limits = meter.limits();
    let classical = super::super::ClassicalTransportQuery {
        outcomes: Arc::clone(call.outcomes),
        treatments: Arc::clone(call.treatments),
        source: Arc::clone(call.anchor),
        target: Arc::clone(call.target),
    };
    // A budget or cancellation observed at engine construction stops the search
    // before any recursive accounting is taken, so the receipt reports the
    // limits in force with no consumed counts rather than fabricating them.
    let mut engine = match super::super::Engine::new_metered(diagram, &classical, meter, ctx) {
        Ok(engine) => engine,
        Err(error) if error.is_budget_or_cancel() => {
            *receipt = Some(ZTransportLimitsReceipt {
                budget: pre_search_budget_kind(&error),
                steps_limit: limits.steps,
                depth_limit: limits.depth,
                steps_consumed: None,
                depth_reached: None,
                explored_rules: Vec::new(),
            });
            return Err(error);
        }
        Err(error) => return Err(error),
    };
    let initial = match engine.initial() {
        Ok(initial) => initial,
        Err(error) if error.is_budget_or_cancel() => {
            *receipt = Some(search_limits_receipt(&error, &engine, limits, Vec::new()));
            return Err(error);
        }
        Err(error) => return Err(error),
    };
    let mut trace = Vec::new();
    let mut terminal_failure = None;
    let mut unassigned = None;
    let irrelevant = BitSet::with_len(diagram.causal_graph().node_count());
    let outcome = search_trz_state(
        &mut engine,
        &initial,
        call,
        None,
        &[],
        &[],
        &irrelevant,
        0,
        &mut trace,
        &mut terminal_failure,
        &mut unassigned,
    );
    match outcome {
        Ok(result) => {
            let steps_explored = engine.steps;
            let depth_reached = engine.max_depth;
            Ok(TrzSearchResult {
                cross_source_joint: engine.cross_source_joint,
                identified: result.map(|root| (engine.arena, root, trace.clone())),
                terminal_failure,
                unassigned,
                explored_rules: trace,
                steps_explored,
                depth_reached,
            })
        }
        Err(error) if error.is_budget_or_cancel() => {
            *receipt = Some(search_limits_receipt(&error, &engine, limits, trace));
            Err(error)
        }
        Err(error) => Err(error),
    }
}

/// Classify a budget or cancellation observed before the recursive search runs.
pub(super) fn pre_search_budget_kind(error: &IdentificationError) -> ZTransportBudgetKind {
    match error {
        IdentificationError::Cancelled => ZTransportBudgetKind::Cancelled,
        IdentificationError::Budget { budget: IdentificationBudget::Memory } => {
            ZTransportBudgetKind::Memory
        }
        _ => ZTransportBudgetKind::Steps,
    }
}

/// Build a receipt from the engine's own accounting when a budget or
/// cancellation stopped the recursive search. The step-versus-depth
/// distinction is read from the accounting, never guessed.
fn search_limits_receipt(
    error: &IdentificationError,
    engine: &super::super::Engine<'_>,
    limits: super::super::SidLimits,
    explored_rules: Vec<String>,
) -> ZTransportLimitsReceipt {
    let budget = match error {
        IdentificationError::Cancelled => ZTransportBudgetKind::Cancelled,
        IdentificationError::Budget { budget: IdentificationBudget::Memory } => {
            ZTransportBudgetKind::Memory
        }
        _ if engine.max_depth >= limits.depth => ZTransportBudgetKind::Depth,
        _ => ZTransportBudgetKind::Steps,
    };
    ZTransportLimitsReceipt {
        budget,
        steps_limit: limits.steps,
        depth_limit: limits.depth,
        steps_consumed: Some(engine.steps),
        depth_reached: Some(engine.max_depth),
        explored_rules,
    }
}

#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    reason = "Mirrors the paper rule state explicitly"
)]
fn search_trz_state(
    engine: &mut super::super::Engine<'_>,
    state: &super::super::State,
    call: &TrzCall<'_>,
    active: Option<usize>,
    remaining_controllable: &[VariableId],
    active_interventions: &[antecedent_expr::InterventionAssignment],
    irrelevant: &BitSet,
    depth: usize,
    trace: &mut Vec<String>,
    terminal_failure: &mut Option<TrzTerminalFailure>,
    unassigned: &mut Option<VariableId>,
) -> Result<Option<ExprId>, IdentificationError> {
    engine.charge(depth)?;
    let mut ws = antecedent_graph::GraphWorkspace::default();
    // `TRz` rule 1: no active treatment coordinates remain.
    if !state.x.any() {
        trace.push("ztr.line1.marginal".into());
        return Ok(Some(
            engine.marginal(state.kernel, &super::super::difference(&state.v, &state.y))?,
        ));
    }
    let ancestors = engine.cached_ancestors(&state.y, &state.v, &mut ws);
    // `TRz` rule 2: restrict to ancestors of the current outcomes.
    if !ancestors.equal_set(&state.v) {
        trace.push("ztr.line2.ancestors".into());
        let next = super::super::State {
            y: state.y.clone(),
            x: super::super::intersection(&state.x, &ancestors),
            v: ancestors.clone(),
            kernel: engine
                .marginal(state.kernel, &super::super::difference(&state.v, &ancestors))?,
        };
        return search_trz_state(
            engine,
            &next,
            call,
            active,
            remaining_controllable,
            active_interventions,
            irrelevant,
            depth + 1,
            trace,
            terminal_failure,
            unassigned,
        );
    }
    let mut enlarged = super::super::difference(&state.v, &state.x);
    let bar = engine.prepared.ancestors_bar_x(&state.y, &state.v, &state.x, &mut ws);
    enlarged.difference_with(&bar);
    // `TRz` rule 3: add non-treatment vertices outside An(Y) in D_X. The child
    // is constant in them, so a later exchange may fix them at a declared level.
    if enlarged.any() {
        trace.push("ztr.line3.enlarge".into());
        let mut next = state.clone();
        next.x.union_with(&enlarged);
        let mut fixed = irrelevant.clone();
        fixed.union_with(&enlarged);
        let Some(child) = search_trz_state(
            engine,
            &next,
            call,
            active,
            remaining_controllable,
            active_interventions,
            &fixed,
            depth + 1,
            trace,
            terminal_failure,
            unassigned,
        )?
        else {
            return Ok(None);
        };
        // Rule 3 is the identity: P_x(y) = P_{x,w}(y). The child names a
        // coordinate of `w` only where a later exchange fixed it at its
        // declared level, so it is returned unchanged; a child that still
        // reads `w` is averaged over the kernel's own law of `w` given `x`,
        // which is zero exactly where the child conditions on a null event.
        let enlarged_vars = engine.vars(&enlarged)?;
        let free = engine.arena.free_variables(child);
        if enlarged_vars.iter().any(|variable| free.contains(variable)) {
            trace.push("ztr.line3.kernel_weighted".into());
            return Ok(Some(engine.enlarge_output(state, &enlarged, child)?));
        }
        return Ok(Some(child));
    }
    let districts = engine.prepared.c_components(&super::super::difference(&state.v, &state.x));
    // `TRz` rule 4: factor over multiple districts in D \ X.
    if districts.len() > 1 {
        trace.push("ztr.line4.districts".into());
        let mut expressions = Vec::with_capacity(districts.len());
        for district in districts {
            let next = super::super::State {
                y: district.clone(),
                x: super::super::difference(&state.v, &district),
                ..state.clone()
            };
            let Some(child) = search_trz_state(
                engine,
                &next,
                call,
                active,
                remaining_controllable,
                active_interventions,
                irrelevant,
                depth + 1,
                trace,
                terminal_failure,
                unassigned,
            )?
            else {
                return Ok(None);
            };
            expressions.push(child);
        }
        let product = engine.product(expressions);
        return Ok(Some(engine.marginal(
            product,
            &super::super::difference(&super::super::difference(&state.v, &state.x), &state.y),
        )?));
    }
    let district = districts
        .first()
        .ok_or(IdentificationError::InvariantViolated { message: "z_transport.empty_district" })?;
    let containing = engine.prepared.c_components(&state.v);
    // `TRz` rules 5–6 establish C0 and whether D is a single c-component.
    if containing.len() > 1 {
        if containing.iter().any(|d| d.equal_set(district)) {
            // `TRz` rule 7: C0 is a c-component of D, so its kernel is available.
            trace.push("ztr.line7.factor".into());
            let kernel = engine.factor(state, district)?;
            return Ok(Some(
                engine.marginal(kernel, &super::super::difference(district, &state.y))?,
            ));
        }
        let larger = containing.iter().find(|d| district.is_subset_of(d)).ok_or(
            IdentificationError::InvariantViolated {
                message: "z_transport.missing_containing_district",
            },
        )?;
        let next = super::super::State {
            y: state.y.clone(),
            x: super::super::intersection(&state.x, larger),
            v: larger.clone(),
            kernel: engine.factor(state, larger)?,
        };
        // `TRz` rule 8: recurse into the containing c-component of D.
        trace.push("ztr.line8.recurse".into());
        return search_trz_state(
            engine,
            &next,
            call,
            active,
            remaining_controllable,
            active_interventions,
            irrelevant,
            depth + 1,
            trace,
            terminal_failure,
            unassigned,
        );
    }

    // `TRz` line 10 (`TR^mz` line 10 with several sources): with no active
    // experiment every source is a candidate; once a subcall exchanged into one
    // source, only that source's remaining controllables may be exchanged, so a
    // c-factor never mixes sources.
    let candidates = active.map_or_else(|| (0..call.domains.len()).collect(), |i| vec![i]);
    let mut eligibility = Vec::with_capacity(candidates.len());
    for index in candidates {
        let domain = &call.domains[index];
        let controllable =
            if active == Some(index) { remaining_controllable } else { domain.controllable };
        let mut activated = BitSet::with_len(engine.diagram.causal_graph().node_count());
        for variable in controllable.iter().copied() {
            let dense = engine.prepared.var_to_dense(variable)?;
            if state.x.contains(dense) {
                activated.insert(dense);
            }
        }
        let separated = engine.admissible_selections(state, &[], domain.selection_targets)?;
        eligibility.push((index, controllable, activated, separated));
    }
    if !eligibility.iter().any(|(_, _, activated, separated)| activated.any() && *separated) {
        // `TRz` rule 11: rule 10 cannot exchange an active experiment.
        let single = call.domains.len() == 1;
        if let (false, Some(current)) = (single, active) {
            // Another source could exchange here, but only jointly with the
            // active source's experiment: the joint `TR^mz` never fabricates.
            for (index, domain) in call.domains.iter().enumerate() {
                if index == current || engine.cross_source_joint {
                    continue;
                }
                let mut activated = false;
                for variable in domain.controllable.iter().copied() {
                    activated |= state.x.contains(engine.prepared.var_to_dense(variable)?);
                }
                engine.cross_source_joint = activated
                    && engine.admissible_selections(state, &[], domain.selection_targets)?;
            }
        }
        let replace = terminal_failure.as_ref().is_none_or(|recorded| {
            // With several sources, prefer a terminal reached with no active
            // experiment: only that state is forced on every search path.
            !single && !recorded.active_interventions.is_empty() && active_interventions.is_empty()
        });
        if replace {
            let mut c0 = engine.vars(district)?.iter().map(|v| v.raw()).collect::<Vec<_>>();
            c0.sort_unstable();
            let raw_sorted = |engine: &super::super::Engine<'_>, set: &BitSet| {
                engine.vars(set).map(|vars| {
                    let mut raw = vars.iter().map(|v| v.raw()).collect::<Vec<_>>();
                    raw.sort_unstable();
                    raw
                })
            };
            trace.push("ztr.line11.fail".into());
            let (remaining, candidate_active, selection_separated, sources) = if single {
                let (_, controllable, activated, separated) = &eligibility[0];
                (
                    controllable.iter().map(|v| v.raw()).collect(),
                    raw_sorted(engine, activated)?,
                    *separated,
                    Vec::new(),
                )
            } else {
                let mut sources = Vec::with_capacity(eligibility.len());
                for (index, controllable, activated, separated) in &eligibility {
                    sources.push(TrzSourceTerminal {
                        population: call.domains[*index].population.to_string(),
                        remaining_controllable: controllable.iter().map(|v| v.raw()).collect(),
                        candidate_active: raw_sorted(engine, activated)?,
                        selection_separated: *separated,
                    });
                }
                (Vec::new(), Vec::new(), false, sources)
            };
            *terminal_failure = Some(TrzTerminalFailure {
                outcomes: engine.vars(&state.y)?.iter().map(|v| v.raw()).collect(),
                treatments: engine.vars(&state.x)?.iter().map(|v| v.raw()).collect(),
                vertices: engine.vars(&state.v)?.iter().map(|v| v.raw()).collect(),
                c0,
                remaining_controllable: remaining,
                candidate_active,
                active_interventions: active_interventions
                    .iter()
                    .map(|a| (a.variable.raw(), a.value.as_f64()))
                    .collect(),
                selection_separated,
                rules: trace.clone(),
                sources,
            });
        }
        return Ok(None);
    }
    let eligible = eligibility
        .into_iter()
        .filter(|(_, _, activated, separated)| activated.any() && *separated)
        .collect::<Vec<_>>();
    for (position, (index, controllable, activated, _)) in eligible.iter().enumerate() {
        match exchange_trz_source(
            engine,
            state,
            call,
            *index,
            controllable,
            activated,
            active_interventions,
            irrelevant,
            depth,
            trace,
            terminal_failure,
            unassigned,
        ) {
            Ok(Some(result)) => return Ok(Some(result)),
            Ok(None) => {}
            Err(error) => {
                // A stop leaves the later line-10 source branches untried;
                // name them by population so the record is order-invariant.
                if call.domains.len() > 1 && error.is_budget_or_cancel() {
                    for (later, ..) in &eligible[position + 1..] {
                        let population = call.domains[*later].population;
                        engine.defer(|| format!("multi:line10:source={population}@{depth}"));
                    }
                }
                return Err(error);
            }
        }
    }
    Ok(None)
}

/// `TRz` rule 10 for one source: exchange its active controllables for that
/// source's experiment and recurse with the source active.
#[allow(clippy::too_many_arguments, reason = "The recursion state of `search_trz_state`")]
fn exchange_trz_source(
    engine: &mut super::super::Engine<'_>,
    state: &super::super::State,
    call: &TrzCall<'_>,
    index: usize,
    controllable: &[VariableId],
    activated: &BitSet,
    active_interventions: &[antecedent_expr::InterventionAssignment],
    irrelevant: &BitSet,
    depth: usize,
    trace: &mut Vec<String>,
    terminal_failure: &mut Option<TrzTerminalFailure>,
    unassigned: &mut Option<VariableId>,
) -> Result<Option<ExprId>, IdentificationError> {
    let domain = &call.domains[index];
    let active_vars = engine.vars(activated)?;
    // The level each exchanged coordinate takes in the cited source experiment.
    // A queried treatment and a coordinate that rule 4 handed to this district
    // stay symbolic: the evaluation request binds the former and the enclosing
    // summation binds the latter, so one formula answers every requested level
    // and every summand at its own world. A coordinate that rule 3 proved
    // irrelevant to the child may be fixed at its declared level, since the
    // child is constant in it.
    let mut exchanged = Vec::with_capacity(active_vars.len());
    for variable in active_vars.iter().copied() {
        let dense = engine.prepared.var_to_dense(variable)?;
        if irrelevant.contains(dense) && !call.treatments.contains(&variable) {
            let Some(assignment) = domain
                .experiment_assignment
                .iter()
                .find(|assignment| assignment.variable == variable)
            else {
                *unassigned = Some(variable);
                return Ok(None);
            };
            exchanged.push(antecedent_expr::InterventionAssignment::concrete(
                variable,
                assignment.value.clone(),
            ));
        } else {
            exchanged.push(antecedent_expr::InterventionAssignment::symbolic(variable));
        }
    }
    let exchanged_levels =
        exchanged.iter().map(|a| (a.variable.raw(), a.value.as_f64())).collect::<Vec<_>>();
    if call.domains.len() == 1 {
        trace.push(format!("ztr.line10.source_exchange:{exchanged_levels:?}"));
    } else {
        trace
            .push(format!("ztr.line10.source_exchange:{}:{exchanged_levels:?}", domain.population));
    }
    let mut cumulative = active_interventions.to_vec();
    for assignment in exchanged {
        if !cumulative.iter().any(|a| a.variable == assignment.variable) {
            cumulative.push(assignment);
        }
    }
    // `TRz` rule 10: exchange Z∩X and recurse with Z\X and active set I=Z∩X.
    // The new kernel is the source c-factor Q^s_z[V\Z] = P^s_{z ∪ (V_all\V)}(V\Z):
    // every coordinate outside the current vertex set is intervened, not
    // conditioned on. When the state still spans the whole graph that is the
    // cited source law itself; otherwise the c-factor is identified from that
    // law by the interventional-distribution recursion on the graph with the
    // exchanged coordinates removed (sID^z line 7).
    let remaining = super::super::difference(&state.v, activated);
    let Some(kernel) = source_c_factor(
        engine,
        &remaining,
        &cumulative,
        domain.population,
        call.target,
        depth,
        trace,
    )?
    else {
        return Ok(None);
    };
    let next = super::super::State {
        y: state.y.clone(),
        x: super::super::difference(&state.x, activated),
        v: remaining,
        kernel,
    };
    // Fig. 4 line 10 recurses into `TRz` on the reduced graph and retains
    // experiments over the remaining controllable variables.
    let remaining = controllable
        .iter()
        .copied()
        .filter(|variable| !active_vars.contains(variable))
        .collect::<Vec<_>>();
    search_trz_state(
        engine,
        &next,
        call,
        Some(index),
        &remaining,
        &cumulative,
        irrelevant,
        depth + 1,
        trace,
        terminal_failure,
        unassigned,
    )
}

/// The source c-factor `Q^s_z[remaining]` after exchanging `cumulative`, in the
/// z engine's arena. `None` when the interventional-distribution recursion cannot
/// identify it from the cited source law; that is not an obstruction claim.
fn source_c_factor(
    engine: &mut super::super::Engine<'_>,
    remaining: &BitSet,
    cumulative: &[antecedent_expr::InterventionAssignment],
    source: &Arc<str>,
    target: &Arc<str>,
    depth: usize,
    trace: &mut Vec<String>,
) -> Result<Option<ExprId>, IdentificationError> {
    let graph = engine.diagram.causal_graph();
    let exchanged = cumulative.iter().map(|a| a.variable).collect::<Vec<_>>();
    let remaining_vars = engine.vars(remaining)?;
    // An outside coordinate that is not an ancestor of the remaining set in the
    // graph without the exchanged coordinates neither changes the c-factor
    // when intervened on nor needs to be measured: the source law over the
    // ancestral closure is the marginal the theorem reads. Only the ancestral
    // outside coordinates are intervened on and identified away.
    let mut exchanged_nodes = BitSet::with_len(graph.node_count());
    for variable in &exchanged {
        // A variable outside the graph cannot be anyone's parent.
        if let Ok(dense) = engine.prepared.var_to_dense(*variable) {
            exchanged_nodes.insert(dense);
        }
    }
    let mut closure = BitSet::default();
    antecedent_graph::algo::closure_into(
        graph.node_count(),
        |u| graph.parents(u),
        remaining.iter(),
        &mut closure,
        &mut antecedent_graph::GraphWorkspace::default(),
        |_, parent| !exchanged_nodes.contains(parent),
    );
    let kept = engine.vars(&closure)?;
    let outside = kept.iter().copied().filter(|v| !remaining_vars.contains(v)).collect::<Vec<_>>();
    let population = engine.arena.intern_population(Arc::clone(source));
    if outside.is_empty() {
        let variables = engine.arena.intern_var_set(remaining_vars);
        let conditioned_on = engine.arena.empty_var_set();
        let intervention = crate::intervention_support::intern_assignments(
            &mut engine.arena,
            cumulative.iter().cloned(),
        )?;
        return Ok(Some(engine.arena.intern(ExprNode::Distribution {
            variables,
            conditioned_on,
            intervention,
            domain: DomainRef::Interventional,
            population,
            regime: None,
        })));
    }
    // Identify P^s_{z ∪ outside}(remaining) from P^s_z(kept) on the graph
    // without the exchanged coordinates.
    let mut reduced = antecedent_graph::Admg::empty();
    for variable in &kept {
        reduced.add_node(NodeRef::Static(*variable))?;
    }
    let dense = |variable: VariableId| {
        kept.iter().position(|v| *v == variable).map(|i| DenseNodeId::from_raw(index_u32(i)))
    };
    for (from_index, node) in graph.nodes().iter().enumerate() {
        let NodeRef::Static(from) = node else { continue };
        let Some(reduced_from) = dense(*from) else { continue };
        let from_dense = DenseNodeId::from_raw(index_u32(from_index));
        for to in graph.children(from_dense) {
            if let Some(reduced_to) = engine.prepared.dense_to_var(*to).ok().and_then(dense) {
                reduced.insert_directed(reduced_from, reduced_to)?;
            }
        }
        for other in graph.bidirected_neighbors(from_dense) {
            if other.as_usize() > from_index {
                if let Some(reduced_other) =
                    engine.prepared.dense_to_var(*other).ok().and_then(dense)
                {
                    reduced.insert_bidirected(reduced_from, reduced_other)?;
                }
            }
        }
    }
    let diagram = SelectionDiagram::try_new(reduced, Arc::<[VariableId]>::from([]))?;
    let classical = super::super::ClassicalTransportQuery {
        outcomes: remaining_vars.clone().into(),
        treatments: outside.into(),
        source: Arc::clone(source),
        target: Arc::clone(target),
    };
    // The nested engine charges the enclosing search's budget: the shared one
    // at its true depth, or the steps and depth this engine has left.
    let ctx = engine.ctx;
    let (sub_steps, solved) = {
        let (meter, depth_offset, base_bytes) = engine.nested_meter(depth);
        let mut sub = super::super::Engine::new_metered(&diagram, &classical, meter, ctx)?;
        sub.depth_offset = depth_offset;
        sub.base_bytes = base_bytes;
        let initial = sub.initial_source_kernel(source, cumulative)?;
        let solved = sub.solve(initial, false, 0).map(|root| {
            root.map(|root_step| {
                let rules = format!("ztr.line10.c_factor:{:?}", sub.reachable_rules(root_step));
                (rules, sub.proof[root_step].output, std::mem::take(&mut sub.arena))
            })
        });
        (sub.steps, solved)
    };
    engine.steps = engine.steps.saturating_add(sub_steps);
    let Some((rules, output, arena)) = solved? else {
        trace.push("ztr.line10.c_factor_not_identified".into());
        return Ok(None);
    };
    trace.push(rules);
    let mut memo = std::collections::HashMap::new();
    Ok(Some(transplant(&arena, output, &mut engine.arena, &mut memo)?))
}

/// Copy the expression `id` of `from` into `into`, re-interning every table.
fn transplant(
    from: &CausalExprArena,
    id: ExprId,
    into: &mut CausalExprArena,
    memo: &mut std::collections::HashMap<ExprId, ExprId>,
) -> Result<ExprId, IdentificationError> {
    if let Some(copied) = memo.get(&id) {
        return Ok(*copied);
    }
    let copied = match from.node(id).clone() {
        ExprNode::Distribution {
            variables,
            conditioned_on,
            intervention,
            domain,
            population,
            regime,
        } => {
            let variables = into.intern_var_set(from.var_set(variables).iter().copied());
            let conditioned_on = into.intern_var_set(from.var_set(conditioned_on).iter().copied());
            let intervention = crate::intervention_support::intern_assignments(
                into,
                from.intervention_assignments(intervention).iter().cloned(),
            )?;
            let population = into.intern_population(Arc::from(from.population(population)));
            into.intern(ExprNode::Distribution {
                variables,
                conditioned_on,
                intervention,
                domain,
                population,
                regime,
            })
        }
        ExprNode::Product(list) => {
            let children = from
                .list(list)
                .iter()
                .map(|child| transplant(from, *child, into, memo))
                .collect::<Result<Vec<_>, _>>()?;
            let list = into.intern_list(children);
            into.intern(ExprNode::Product(list))
        }
        ExprNode::SumOut { variables, expr } => {
            let expr = transplant(from, expr, into, memo)?;
            let variables = into.intern_var_set(from.var_set(variables).iter().copied());
            into.intern(ExprNode::SumOut { variables, expr })
        }
        ExprNode::Ratio { numerator, denominator } => {
            let numerator = transplant(from, numerator, into, memo)?;
            let denominator = transplant(from, denominator, into, memo)?;
            into.intern(ExprNode::Ratio { numerator, denominator })
        }
        _ => {
            return Err(IdentificationError::InvariantViolated {
                message: "source c-factor identification produced an unexpected node",
            });
        }
    };
    memo.insert(id, copied);
    Ok(copied)
}
