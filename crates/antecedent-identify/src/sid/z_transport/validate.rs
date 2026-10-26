//! Validate items of the parent module.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::IdentificationError;
use antecedent_core::{DistributionAvailability, EvidenceCatalog, RegimeKind, VariableDomain, VariableId};
use antecedent_graph::{NodeRef, SelectionDiagram};
use super::search::same_variable_set;

use super::*;

/// Validate the graph/query portion of the bounded z-transport contract.
///
/// The number of observed variables is not bounded here: the search that consumes the
/// query charges its steps, depth, memory and cancellation to a budget, and a stop is a
/// limits receipt, never a verdict. This check does not imply that experiments exist.
/// Callers must bind every formula factor to an available exact joint regime with
/// matching population, measurement scope, intervention set, and intervention values.
///
/// # Errors
/// Returns a stable validation error when the graph, query, population identities,
/// or declared controllable bound is invalid.
pub fn validate_z_transport_query(
    diagram: &SelectionDiagram,
    query: &ZTransportQuery,
) -> Result<(), IdentificationError> {
    if query.controllable.is_empty() {
        return Err(IdentificationError::UnsupportedInput {
            code: "z_transport.unsupported_controllable_count",
        });
    }
    if query.outcomes.is_empty()
        || query.treatments.is_empty()
        || query.source.trim().is_empty()
        || query.target.trim().is_empty()
        || query.source == query.target
    {
        return Err(IdentificationError::invalid_input("z_transport.invalid_query"));
    }

    for group in [&query.outcomes, &query.treatments, &query.controllable] {
        let mut seen = std::collections::BTreeSet::new();
        for variable in group.iter().copied() {
            if !diagram.causal_graph().nodes().contains(&NodeRef::Static(variable)) {
                return Err(IdentificationError::invalid_input("z_transport.unknown_variable"));
            }
            if !seen.insert(variable.raw()) {
                return Err(IdentificationError::invalid_input("z_transport.duplicate_variable"));
            }
        }
    }
    if query.outcomes.iter().any(|v| query.treatments.contains(v)) {
        return Err(IdentificationError::invalid_input("z_transport.outcomes_overlap_treatments"));
    }
    let mut assigned = std::collections::BTreeSet::new();
    for assignment in query.experiment_assignment.iter() {
        if !query.controllable.contains(&assignment.variable)
            || !assigned.insert(assignment.variable.raw())
            || assignment.value.validate_concrete_intervention_level().is_err()
        {
            return Err(IdentificationError::invalid_input(
                "z_transport.invalid_experiment_assignment",
            ));
        }
    }
    Ok(())
}

/// Check for the finite full-law experiment family declared by the zTR contract.
///
/// The theorem input family includes a joint source law for every joint assignment
/// to each non-empty subset of controllable variables. An observational law is not
/// part of this family unless a separate theorem premise requires it.
/// Results here are catalog facts only; a successful check does not identify a
/// query or establish that one supplied regime can substitute for another.
#[allow(
    clippy::float_cmp,
    reason = "Categorical intervention labels are finite exact integers encoded as f64"
)]
#[allow(
    clippy::too_many_lines,
    reason = "one linear identification pipeline whose stages share the prepared graph and workspace"
)]
pub fn validate_z_experiment_family(
    diagram: &SelectionDiagram,
    query: &ZTransportQuery,
    catalog: &EvidenceCatalog,
) -> Result<Vec<antecedent_core::RegimeId>, ZExperimentFamilyError> {
    let source =
        catalog.environments.iter().find(|environment| environment.identity == query.source);
    let target =
        catalog.environments.iter().find(|environment| environment.identity == query.target);
    let observed = diagram
        .causal_graph()
        .nodes()
        .iter()
        .filter_map(|node| match node {
            NodeRef::Static(variable) => Some(*variable),
            _ => None,
        })
        .collect::<Vec<_>>();
    for variable in &observed {
        for environment in [source, target].into_iter().flatten() {
            let domain = environment
                .variables
                .iter()
                .find(|coordinate| coordinate.variable == *variable)
                .map(|coordinate| &coordinate.domain);
            if !matches!(domain, Some(VariableDomain::Binary | VariableDomain::Categorical { .. }))
            {
                return Err(ZExperimentFamilyError::UnsupportedDomain { variable: *variable });
            }
        }
    }
    let domains = query
        .controllable
        .iter()
        .map(|variable| {
            let coordinate = source.and_then(|environment| {
                environment.variables.iter().find(|c| c.variable == *variable)
            });
            let domain = coordinate.map(|coordinate| &coordinate.domain);
            let values = match domain {
                Some(VariableDomain::Binary) => vec![0.0, 1.0],
                Some(VariableDomain::Categorical { cardinality }) => {
                    (0..*cardinality).map(f64::from).collect()
                }
                _ => return Err(ZExperimentFamilyError::UnsupportedDomain { variable: *variable }),
            };
            Ok((*variable, values))
        })
        .collect::<Result<Vec<_>, _>>()?;
    // Every non-empty subset at every joint level: prod(1 + levels) - 1 regimes, enumerated
    // lazily in a fixed order. Each required regime must be a distinct catalog regime, so
    // the first absent one is reached after at most `catalog.regimes.len() + 1` regimes
    // however wide the controllable set: the family is never materialized.
    let mut required = Vec::new();
    let k = query.controllable.len();
    let masks = if k < 64 { (1u64 << k) - 1 } else { u64::MAX };
    for mask in 1..=masks {
        let intervened = (0..k.min(64))
            .filter(|bit| (mask >> bit) & 1 == 1)
            .map(|bit| query.controllable[bit])
            .collect::<Vec<_>>();
        let level_sets = intervened
            .iter()
            .map(|variable| {
                &domains
                    .iter()
                    .find(|(candidate, _)| candidate == variable)
                    .expect("controllable coordinate was enumerated")
                    .1
            })
            .collect::<Vec<_>>();
        if level_sets.iter().any(|levels| levels.is_empty()) {
            continue;
        }
        // Mixed-radix walk, first variable most significant.
        let mut digits = vec![0usize; intervened.len()];
        loop {
            let assignment = intervened
                .iter()
                .zip(&level_sets)
                .zip(&digits)
                .map(|((variable, levels), digit)| (*variable, levels[*digit]))
                .collect::<Vec<(VariableId, f64)>>();
            let supplied = catalog.regimes.iter().find(|regime| {
                let kind_matches = if intervened.is_empty() {
                    regime.kind == RegimeKind::Observational
                } else {
                    regime.kind == RegimeKind::Experimental
                };
                let regime_values = regime
                    .intervention_values
                    .iter()
                    .filter_map(|value| value.value.as_f64().map(|v| (value.variable, v)))
                    .collect::<Vec<_>>();
                kind_matches
                    && regime.supplies_population_law()
                    && regime.population.as_ref() == query.source.as_ref()
                    && same_variable_set(&regime.interventions, &intervened)
                    && (regime_values.len() == assignment.len()
                        && assignment.iter().all(|(variable, value)| {
                            regime_values.iter().any(|(v, actual)| v == variable && actual == value)
                        }))
                    && same_variable_set(&regime.measured, &observed)
                    && regime.conditioned_on.is_empty()
                    && regime.distribution == DistributionAvailability::Joint
                    && catalog.bindings.iter().any(|binding| binding.regime == regime.id)
            });
            if let Some(regime) = supplied {
                required.push(regime.id);
            } else {
                return Err(ZExperimentFamilyError::MissingJointLaw {
                    interventions: intervened.into(),
                    values: assignment.into(),
                });
            }
            let mut place = digits.len();
            let mut advanced = false;
            while place > 0 {
                place -= 1;
                digits[place] += 1;
                if digits[place] < level_sets[place].len() {
                    advanced = true;
                    break;
                }
                digits[place] = 0;
            }
            if !advanced {
                break;
            }
        }
    }
    Ok(required)
}
