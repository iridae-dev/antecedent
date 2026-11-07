//! Bind z-transport expressions to available catalog regimes.

use super::{
    Arc, BoundZTransportFunctional, CausalExprArena, DistributionAvailability, EvidenceCatalog,
    ExprId, ExprNode, IdentificationBudget, IdentificationError, RegimeKind, SelectionDiagram,
    VariableDomain, VariableId, ZFormulaKind, ZTransportDerivation, ZTransportQuery,
    same_variable_set,
};

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
pub(super) fn regime_matches_world(
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
        if worlds.len().saturating_mul(levels.len()) > MAX_BOUND_WORLDS {
            return Err(IdentificationError::budget(IdentificationBudget::ZTransport));
        }
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

/// Most worlds one symbolic leaf binding enumerates (a cartesian product of declared levels).
const MAX_BOUND_WORLDS: usize = 65_536;

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
