//! Detached finite-discrete replay for checked functional artifacts.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{RegimeId, Value, VariableId};
use antecedent_expr::{
    Assignment, DistributionProvider, DomainRef, EmpiricalTableProvider, EvalContext, EvalError,
    FactorRequirement, FactorSpec, FunctionalProgram, InterventionAssignment,
};

use crate::contract_reason::ContractReason;
use crate::{
    CausalQueryWire, DistributionFactorDomainWire, DistributionFactorKeyWire,
    DistributionFactorLawsWire, DistributionFactorTableWire, FunctionalProgramWire,
    InterventionWire, InterventionalDistributionWire, ValueWire, functional_program_from_wire,
};

/// Wire format version of the portable factor laws.
pub const FACTOR_LAWS_FORMAT: u16 = 1;
const MASS_TOLERANCE: f64 = 1e-9;
/// Default most table cells (domain levels plus factor entries) a replay admits.
pub const MAX_REPLAY_CELLS: usize = 1_000_000;
/// Default most evaluator operations (support cells times program nodes) a replay performs.
pub const MAX_REPLAY_OPERATIONS: usize = 50_000_000;

/// Resource bounds of the detached finite-discrete replay of a checked functional artifact.
/// The defaults are [`MAX_REPLAY_CELLS`] and [`MAX_REPLAY_OPERATIONS`]; a consumer with a
/// larger budget passes its own to [`crate::contract_section::verify_contract_against_body_within`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReplayLimits {
    /// Most table cells a replay admits.
    pub max_cells: usize,
    /// Most evaluator operations a replay performs.
    pub max_operations: usize,
}

impl Default for ReplayLimits {
    fn default() -> Self {
        Self { max_cells: MAX_REPLAY_CELLS, max_operations: MAX_REPLAY_OPERATIONS }
    }
}

/// Recompute every finite-discrete distribution atom from its checked program and portable
/// factor laws. Returns a stable dependency/error key suitable for a refusal report.
#[allow(
    clippy::too_many_lines,
    reason = "The bounded joint atom enumeration and its validation must share one replay context."
)]
pub(crate) fn replay_distribution_atoms(
    query: &CausalQueryWire,
    result: &InterventionalDistributionWire,
    program_wire: &FunctionalProgramWire,
    laws: &DistributionFactorLawsWire,
    limits: &ReplayLimits,
) -> Result<(), ContractReason> {
    let CausalQueryWire::Distribution(query) = query else {
        return Err(ContractReason::DistributionQueryKind);
    };
    if !matches!(query.target_population, crate::TargetPopulationWire::AllObserved) {
        return Err(ContractReason::DistributionPopulation);
    }
    if laws.format != FACTOR_LAWS_FORMAT {
        return Err(ContractReason::DistributionFactorLawsFormat);
    }
    if laws.provider != "empirical_table"
        || laws.missing_row_policy != "joint_complete_case"
        || laws.complete_case_rows == 0
        || laws.source_rows < laws.complete_case_rows
    {
        return Err(ContractReason::DistributionProviderProvenance);
    }
    let program =
        functional_program_from_wire(program_wire, antecedent_expr::ProgramLimits::default())
            .map_err(|_| ContractReason::DistributionFunctionalProgram)?;
    verify_requirements(&program, query, laws)?;
    validate_law_size(laws, limits)?;
    let atom_width = query.outcomes.len().saturating_add(query.conditioning.len()).max(1);
    let atom_scans =
        result.atoms.len().saturating_mul(result.atoms.len()).saturating_mul(atom_width);
    let scan_work = law_scan_work(laws).saturating_add(atom_scans);
    if scan_work == usize::MAX || scan_work > limits.max_operations {
        return Err(ContractReason::DistributionReplayResourceLimit);
    }
    let provider = provider_from_laws(laws)?;

    let outcomes: Vec<_> = query.outcomes.iter().copied().map(VariableId::from_raw).collect();
    let conditioning: Vec<_> =
        query.conditioning.iter().copied().map(VariableId::from_raw).collect();
    let interventions = query
        .interventions
        .iter()
        .map(|intervention| match intervention {
            InterventionWire::Set { variable, value } => {
                Ok((VariableId::from_raw(*variable), value.to_value()))
            }
            _ => Err(ContractReason::DistributionInterventionKind),
        })
        .collect::<Result<Vec<_>, _>>()?;
    let evaluator = program.compile().map_err(|_| ContractReason::DistributionProgramCompile)?;
    let y_support = provider
        .support(&outcomes, &EvalContext::default())
        .map_err(|_| ContractReason::DistributionOutcomeSupport)?;
    let z_support = provider
        .support(&conditioning, &EvalContext::default())
        .map_err(|_| ContractReason::DistributionConditioningSupport)?;
    let bound: Vec<_> = outcomes
        .iter()
        .copied()
        .chain(conditioning.iter().copied())
        .chain(interventions.iter().map(|(variable, _)| *variable))
        .collect();
    let free: Vec<_> = program
        .free_variables()
        .iter()
        .copied()
        .filter(|variable| !bound.contains(variable))
        .collect();
    let outcome_cells = support_cells(laws, &outcomes, limits)?;
    let conditioning_cells = support_cells(laws, &conditioning, limits)?;
    let free_cells = support_cells(laws, &free, limits)?;
    let max_program_nodes = program.arena().len().max(1);
    if outcome_cells
        .checked_mul(conditioning_cells)
        .and_then(|n| n.checked_mul(free_cells))
        .and_then(|n| n.checked_mul(max_program_nodes))
        .and_then(|n| n.checked_add(scan_work))
        .is_none_or(|n| n > limits.max_operations)
    {
        return Err(ContractReason::DistributionReplayResourceLimit);
    }
    for (variable, value) in &interventions {
        let domain = laws
            .domains
            .iter()
            .find(|(id, _)| *id == variable.raw())
            .ok_or(ContractReason::DistributionFactorDomains)?;
        if !domain.1.iter().any(|level| level.to_value() == *value) {
            return Err(ContractReason::DistributionInterventionOutsideDomain);
        }
    }
    let free_support = provider
        .support(&free, &EvalContext::default())
        .map_err(|_| ContractReason::DistributionFreeVariableSupport)?;
    let free_law = if free.is_empty() {
        None
    } else {
        Some(FactorSpec::new(&free, &[], &[], DomainRef::Observational))
    };

    let mut expected = Vec::with_capacity(y_support.len().saturating_mul(z_support.len()));
    for z_row in z_support.iter() {
        let z_pairs: Vec<_> = conditioning.iter().copied().zip(z_row.iter().cloned()).collect();
        let mut probabilities = vec![0.0; y_support.len()];
        let mut total_weight = 0.0;
        for free_row in free_support.iter() {
            let free_pairs: Vec<_> = free.iter().copied().zip(free_row.iter().cloned()).collect();
            let mut base = Assignment::from_pairs(
                interventions
                    .iter()
                    .cloned()
                    .chain(z_pairs.iter().cloned())
                    .chain(free_pairs.iter().cloned()),
            );
            let weight = match (&free_law, free.is_empty()) {
                (Some(spec), false) => provider
                    .probability(spec, &base, &EvalContext::default())
                    .map_err(|_| ContractReason::DistributionFreeVariableLaw)?,
                _ => 1.0,
            };
            if !weight.is_finite() || weight < 0.0 {
                return Err(ContractReason::DistributionFreeVariableLaw);
            }
            if weight == 0.0 {
                continue;
            }
            let mut row_values = Vec::with_capacity(y_support.len());
            let mut evaluable = true;
            for y_row in y_support.iter() {
                for (variable, value) in outcomes.iter().copied().zip(y_row.iter().cloned()) {
                    base.set(variable, value);
                }
                match evaluator.evaluate_with(&provider, &EvalContext::default(), &base) {
                    Ok(probability)
                        if probability.is_finite() && (0.0..=1.0).contains(&probability) =>
                    {
                        row_values.push(probability)
                    }
                    Ok(_) => return Err(ContractReason::DistributionAtomProbability),
                    Err(
                        EvalError::MissingTableEntry
                        | EvalError::EmptySupport(_)
                        | EvalError::DivisionByZero,
                    ) => {
                        evaluable = false;
                        break;
                    }
                    Err(_) => return Err(ContractReason::DistributionExpressionEvaluation),
                }
                for variable in &outcomes {
                    base.remove(*variable);
                }
            }
            if !evaluable {
                continue;
            }
            for (slot, value) in probabilities.iter_mut().zip(row_values) {
                *slot += weight * value;
            }
            total_weight += weight;
        }
        if total_weight <= 0.0 {
            return Err(ContractReason::DistributionNoEvaluableFreeVariableMass);
        }
        for (y_row, probability) in y_support.iter().zip(probabilities) {
            let outcome_pairs = outcomes
                .iter()
                .copied()
                .zip(y_row.iter().cloned())
                .map(|(id, value)| (id.raw(), ValueWire::from_value(&value)))
                .collect::<Vec<_>>();
            let conditioning_pairs = conditioning
                .iter()
                .copied()
                .zip(z_row.iter().cloned())
                .map(|(id, value)| (id.raw(), ValueWire::from_value(&value)))
                .collect::<Vec<_>>();
            expected.push((outcome_pairs, conditioning_pairs, probability / total_weight));
        }
    }
    if expected.len() != result.atoms.len() {
        return Err(ContractReason::DistributionAtomCount);
    }
    for (outcomes, conditioning, probability) in expected {
        let matches: Vec<_> = result
            .atoms
            .iter()
            .filter(|atom| atom.outcomes == outcomes && atom.conditioning == conditioning)
            .collect();
        if matches.len() != 1 || (matches[0].probability - probability).abs() > MASS_TOLERANCE {
            return Err(ContractReason::DistributionAtomReplayMismatch);
        }
    }
    Ok(())
}

/// Recompute a checked discrete scalar functional from its program and empirical factor laws.
///
/// The returned key names a stable refusal reason and is suitable for an independent
/// consumer's dependency report.
pub(crate) fn replay_functional_scalar(
    query: &CausalQueryWire,
    reported: Option<f64>,
    program_wire: &FunctionalProgramWire,
    laws: &DistributionFactorLawsWire,
    limits: &ReplayLimits,
) -> Result<(), ContractReason> {
    if !matches!(
        query,
        CausalQueryWire::AverageEffect { .. }
            | CausalQueryWire::PathSpecific(_)
            | CausalQueryWire::NestedCounterfactual { .. }
            | CausalQueryWire::Response(_)
    ) {
        return Err(ContractReason::FunctionalEffectQueryKind);
    }
    let reported = reported.ok_or(ContractReason::FunctionalEffectResult)?;
    if !reported.is_finite() {
        return Err(ContractReason::FunctionalEffectResult);
    }
    if laws.format != FACTOR_LAWS_FORMAT {
        return Err(ContractReason::FunctionalEffectFactorLawsFormat);
    }
    if laws.provider != "empirical_table"
        || laws.missing_row_policy != "joint_complete_case"
        || laws.complete_case_rows == 0
        || laws.source_rows < laws.complete_case_rows
    {
        return Err(ContractReason::FunctionalEffectProviderProvenance);
    }
    let program =
        functional_program_from_wire(program_wire, antecedent_expr::ProgramLimits::default())
            .map_err(|_| ContractReason::FunctionalEffectFunctionalProgram)?;
    verify_scalar_requirements(&program, laws)?;
    validate_law_size(laws, limits)
        .map_err(|_| ContractReason::FunctionalEffectFactorLawsResourceLimit)?;
    let scan_work = law_scan_work(laws);
    if scan_work == usize::MAX || scan_work > limits.max_operations {
        return Err(ContractReason::FunctionalEffectReplayResourceLimit);
    }
    let provider = provider_from_laws(laws).map_err(|reason| match reason {
        ContractReason::DistributionFactorNormalization => {
            ContractReason::FunctionalEffectFactorNormalization
        }
        _ => ContractReason::FunctionalEffectFactorLaws,
    })?;
    let free = program.free_variables();
    let free_cells = support_cells(laws, free, limits)
        .map_err(|_| ContractReason::FunctionalEffectFactorDomains)?;
    if free_cells
        .checked_mul(program.arena().len().max(1))
        .and_then(|n| n.checked_add(scan_work))
        .is_none_or(|n| n > limits.max_operations)
    {
        return Err(ContractReason::FunctionalEffectReplayResourceLimit);
    }
    let support = provider
        .support(free, &EvalContext::default())
        .map_err(|_| ContractReason::FunctionalEffectFreeVariableSupport)?;
    let free_law =
        (!free.is_empty()).then(|| FactorSpec::new(free, &[], &[], DomainRef::Observational));
    let evaluator =
        program.compile().map_err(|_| ContractReason::FunctionalEffectProgramCompile)?;
    let mut weighted_sum = 0.0;
    let mut total_weight = 0.0;
    for row in support.iter() {
        let assignment = Assignment::from_pairs(free.iter().copied().zip(row.iter().cloned()));
        let weight = match &free_law {
            Some(spec) => provider
                .probability(spec, &assignment, &EvalContext::default())
                .map_err(|_| ContractReason::FunctionalEffectFreeVariableLaw)?,
            None => 1.0,
        };
        if !weight.is_finite() || weight < 0.0 {
            return Err(ContractReason::FunctionalEffectFreeVariableLaw);
        }
        if weight == 0.0 {
            continue;
        }
        let value = evaluator
            .evaluate_with(&provider, &EvalContext::default(), &assignment)
            .map_err(|_| ContractReason::FunctionalEffectExpressionEvaluation)?;
        if !value.is_finite() {
            return Err(ContractReason::FunctionalEffectNonFiniteValue);
        }
        weighted_sum += weight * value;
        total_weight += weight;
    }
    if !total_weight.is_finite() || total_weight <= 0.0 {
        return Err(ContractReason::FunctionalEffectZeroFreeVariableMass);
    }
    let expected = weighted_sum / total_weight;
    if !expected.is_finite() || (expected - reported).abs() > 1e-10 * (1.0 + expected.abs()) {
        return Err(ContractReason::FunctionalEffectResultMismatch);
    }
    Ok(())
}

fn verify_scalar_requirements(
    program: &FunctionalProgram,
    laws: &DistributionFactorLawsWire,
) -> Result<(), ContractReason> {
    let mut expected: Vec<_> =
        program.factor_requirements().iter().map(key_from_requirement).collect();
    let free: Vec<_> = program.free_variables().iter().map(|id| id.raw()).collect();
    if !free.is_empty() {
        expected.push(DistributionFactorKeyWire {
            variables: free,
            conditioned_on: Vec::new(),
            intervention: Vec::new(),
            domain: DistributionFactorDomainWire::Observational,
            population: String::new(),
            regime: None,
        });
    }
    if expected.iter().any(|requirement| !laws.requirements.contains(requirement)) {
        return Err(ContractReason::FunctionalEffectFactorRequirements);
    }
    let mut unique = Vec::new();
    for key in expected {
        if !unique.contains(&key) {
            unique.push(key);
        }
    }
    if unique.iter().any(|key| !laws.factors.iter().any(|factor| &factor.key == key)) {
        return Err(ContractReason::FunctionalEffectFactorLawCoverage);
    }
    Ok(())
}

fn verify_requirements(
    program: &FunctionalProgram,
    query: &crate::InterventionalDistributionQueryWire,
    laws: &DistributionFactorLawsWire,
) -> Result<(), ContractReason> {
    let mut expected: Vec<_> =
        program.factor_requirements().iter().map(key_from_requirement).collect();
    let bound: Vec<_> = query
        .outcomes
        .iter()
        .chain(query.conditioning.iter())
        .copied()
        .chain(query.interventions.iter().filter_map(|intervention| match intervention {
            InterventionWire::Set { variable, .. } => Some(*variable),
            _ => None,
        }))
        .collect();
    let free: Vec<_> = program
        .free_variables()
        .iter()
        .map(|variable| variable.raw())
        .filter(|variable| !bound.contains(variable))
        .collect();
    if !free.is_empty() {
        expected.push(DistributionFactorKeyWire {
            variables: free,
            conditioned_on: Vec::new(),
            intervention: Vec::new(),
            domain: DistributionFactorDomainWire::Observational,
            population: String::new(),
            regime: None,
        });
    }
    if expected != laws.requirements {
        return Err(ContractReason::DistributionFactorRequirements);
    }
    let mut unique_expected = Vec::new();
    for key in &expected {
        if !unique_expected.contains(key) {
            unique_expected.push(key.clone());
        }
    }
    if unique_expected.len() != laws.factors.len()
        || unique_expected.iter().any(|key| !laws.factors.iter().any(|table| &table.key == key))
        || laws.factors.iter().any(|table| !unique_expected.contains(&table.key))
    {
        return Err(ContractReason::DistributionFactorLawCoverage);
    }
    Ok(())
}

fn key_from_requirement(requirement: &FactorRequirement) -> DistributionFactorKeyWire {
    DistributionFactorKeyWire {
        variables: requirement.variables.iter().map(|id| id.raw()).collect(),
        conditioned_on: requirement.conditioned_on.iter().map(|id| id.raw()).collect(),
        intervention: requirement
            .intervention
            .iter()
            .map(|assignment| crate::expr_wire::InterventionAssignmentWire {
                symbolic: assignment.is_symbolic(),
                variable: assignment.variable.raw(),
                value: ValueWire::from_value(&assignment.value),
            })
            .collect(),
        domain: match requirement.domain {
            DomainRef::Observational => DistributionFactorDomainWire::Observational,
            DomainRef::Interventional => DistributionFactorDomainWire::Interventional,
        },
        population: requirement.population.to_string(),
        regime: requirement.regime.map(RegimeId::raw),
    }
}

fn provider_from_laws(
    laws: &DistributionFactorLawsWire,
) -> Result<EmpiricalTableProvider, ContractReason> {
    let mut provider = EmpiricalTableProvider::new();
    validate_domains(laws, &mut provider)?;
    for table in &laws.factors {
        insert_factor_table(laws, table, &mut provider)?;
    }
    Ok(provider)
}

fn validate_domains(
    laws: &DistributionFactorLawsWire,
    provider: &mut EmpiricalTableProvider,
) -> Result<(), ContractReason> {
    let mut seen_domains = Vec::new();
    for (raw, levels) in &laws.domains {
        let variable = VariableId::from_raw(*raw);
        if levels.is_empty()
            || seen_domains.contains(raw)
            || levels
                .iter()
                .any(|value| matches!(value, ValueWire::Float64(number) if !number.is_finite()))
            || levels.iter().enumerate().any(|(index, value)| levels[..index].contains(value))
        {
            return Err(ContractReason::DistributionFactorDomains);
        }
        seen_domains.push(*raw);
        provider.set_domain(variable, levels.iter().map(ValueWire::to_value));
    }
    Ok(())
}

fn insert_factor_table(
    laws: &DistributionFactorLawsWire,
    table: &DistributionFactorTableWire,
    provider: &mut EmpiricalTableProvider,
) -> Result<(), ContractReason> {
    let key = &table.key;
    let variables: Vec<_> = key.variables.iter().copied().map(VariableId::from_raw).collect();
    let conditioned_on: Vec<_> =
        key.conditioned_on.iter().copied().map(VariableId::from_raw).collect();
    let intervention: Vec<_> = key
        .intervention
        .iter()
        .map(|assignment| {
            if assignment.symbolic {
                InterventionAssignment::symbolic(VariableId::from_raw(assignment.variable))
            } else {
                InterventionAssignment::concrete(
                    VariableId::from_raw(assignment.variable),
                    assignment.value.to_value(),
                )
            }
        })
        .collect();
    if intervention.iter().any(InterventionAssignment::is_symbolic) {
        return Err(ContractReason::DistributionSymbolicFactorIntervention);
    }
    let spec = FactorSpec {
        variables: &variables,
        conditioned_on: &conditioned_on,
        intervention: &intervention,
        domain: match key.domain {
            DistributionFactorDomainWire::Observational => DomainRef::Observational,
            DistributionFactorDomainWire::Interventional => DomainRef::Interventional,
        },
        population: &key.population,
        regime: key.regime.map(RegimeId::from_raw),
    };
    let expected_cells = variables
        .iter()
        .chain(conditioned_on.iter())
        .try_fold(1usize, |product, id| {
            laws.domains
                .iter()
                .find(|(raw, _)| *raw == id.raw())
                .map(|(_, levels)| product.saturating_mul(levels.len()))
        })
        .ok_or(ContractReason::DistributionFactorDomains)?;
    if table.rows.len() != expected_cells {
        return Err(ContractReason::DistributionFactorTableIncomplete);
    }
    let mut assignments_seen: Vec<Vec<Value>> = Vec::new();
    let mut conditional_masses: Vec<(Vec<Value>, f64)> = Vec::new();
    for row in &table.rows {
        if row.values.len() != variables.len() + conditioned_on.len()
            || !row.probability.is_finite()
            || !(0.0..=1.0).contains(&row.probability)
        {
            return Err(ContractReason::DistributionFactorRow);
        }
        let values: Vec<_> = row.values.iter().map(ValueWire::to_value).collect();
        if assignments_seen.contains(&values) {
            return Err(ContractReason::DistributionFactorDuplicateRow);
        }
        for (variable, value) in variables.iter().chain(conditioned_on.iter()).zip(values.iter()) {
            let domain = laws
                .domains
                .iter()
                .find(|(raw, _)| *raw == variable.raw())
                .ok_or(ContractReason::DistributionFactorDomains)?;
            if !domain.1.iter().any(|level| level.to_value() == *value) {
                return Err(ContractReason::DistributionFactorRowOutsideDomain);
            }
        }
        assignments_seen.push(values.clone());
        let condition = values[variables.len()..].to_vec();
        if let Some((_, mass)) =
            conditional_masses.iter_mut().find(|(existing, _)| *existing == condition)
        {
            *mass += row.probability;
        } else {
            conditional_masses.push((condition, row.probability));
        }
        let assignment = Assignment::from_pairs(
            variables.iter().chain(conditioned_on.iter()).copied().zip(values),
        );
        provider
            .insert_probability(&spec, &assignment, row.probability)
            .map_err(|_| ContractReason::DistributionFactorRow)?;
    }
    if conditional_masses.iter().any(|(_, mass)| (mass - 1.0).abs() > MASS_TOLERANCE) {
        return Err(ContractReason::DistributionFactorNormalization);
    }
    Ok(())
}

// Conservative comparison bounds for the linear scans performed while building the
// empirical provider. Saturation turns an unrepresentable bound into a refusal.
fn law_scan_work(laws: &DistributionFactorLawsWire) -> usize {
    let domain_count = laws.domains.len();
    let max_levels = laws.domains.iter().map(|(_, levels)| levels.len()).max().unwrap_or(0);
    let mut work = domain_count.saturating_mul(domain_count);
    for (_, levels) in &laws.domains {
        work = work.saturating_add(levels.len().saturating_mul(levels.len()));
    }
    for table in &laws.factors {
        let width = table.key.variables.len().saturating_add(table.key.conditioned_on.len());
        let rows = table.rows.len();
        // Duplicate assignments and condition groups each scan prior rows. Comparing a
        // row or condition may examine every value. Domain lookups and level membership
        // are repeated for every row and column; table shape also looks up each domain.
        let duplicates = rows.saturating_mul(rows).saturating_mul(width.max(1));
        let groups =
            rows.saturating_mul(rows).saturating_mul(table.key.conditioned_on.len().max(1));
        let domain_lookups = rows
            .saturating_add(1)
            .saturating_mul(width)
            .saturating_mul(domain_count.saturating_add(max_levels));
        work =
            work.saturating_add(duplicates).saturating_add(groups).saturating_add(domain_lookups);
    }
    work
}

fn validate_law_size(
    laws: &DistributionFactorLawsWire,
    limits: &ReplayLimits,
) -> Result<(), ContractReason> {
    let domain_entries =
        laws.domains.iter().fold(0usize, |total, (_, levels)| total.saturating_add(levels.len()));
    let factor_cells = laws.factors.iter().fold(0usize, |total, factor| {
        total.saturating_add(factor.rows.iter().fold(0usize, |subtotal, row| {
            subtotal.saturating_add(row.values.len().saturating_add(1))
        }))
    });
    if domain_entries.saturating_add(factor_cells) > limits.max_cells {
        return Err(ContractReason::DistributionReplayResourceLimit);
    }
    Ok(())
}

fn support_cells(
    laws: &DistributionFactorLawsWire,
    variables: &[VariableId],
    limits: &ReplayLimits,
) -> Result<usize, ContractReason> {
    let count = variables
        .iter()
        .try_fold(1usize, |product, variable| {
            laws.domains
                .iter()
                .find(|(raw, _)| *raw == variable.raw())
                .map(|(_, levels)| product.saturating_mul(levels.len()))
        })
        .ok_or(ContractReason::DistributionFactorDomains)?;
    if count > limits.max_cells {
        return Err(ContractReason::DistributionReplayResourceLimit);
    }
    Ok(count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contract_reason::ContractReason;
    use crate::{DistributionFactorRowWire, DistributionFactorTableWire};

    fn replay_distribution_atoms(
        query: &CausalQueryWire,
        result: &InterventionalDistributionWire,
        program: &FunctionalProgramWire,
        laws: &DistributionFactorLawsWire,
    ) -> Result<(), ContractReason> {
        super::replay_distribution_atoms(query, result, program, laws, &ReplayLimits::default())
    }

    fn replay_functional_scalar(
        query: &CausalQueryWire,
        reported: Option<f64>,
        program: &FunctionalProgramWire,
        laws: &DistributionFactorLawsWire,
    ) -> Result<(), ContractReason> {
        super::replay_functional_scalar(query, reported, program, laws, &ReplayLimits::default())
    }
    use antecedent_expr::{
        CausalExprArena, FunctionalProgram, InterventionAssignment, ProgramLimits, ProgramSchema,
        ProgramVariable,
    };
    use std::sync::Arc;

    fn replay_fixture() -> (
        CausalQueryWire,
        InterventionalDistributionWire,
        FunctionalProgramWire,
        DistributionFactorLawsWire,
    ) {
        let treatment = VariableId::from_raw(0);
        let outcome = VariableId::from_raw(1);
        let intervention = InterventionAssignment::concrete(treatment, Value::f64(1.0));
        let mut arena = CausalExprArena::new();
        let y = arena.intern_var_set([outcome]);
        let empty = arena.empty_var_set();
        let do_x = arena.intern_intervention_assignments([intervention.clone()]);
        let root = arena.intern_distribution(y, empty, do_x, DomainRef::Interventional);
        let schema = ProgramSchema::new([
            (treatment, ProgramVariable { name: Arc::from("x") }),
            (outcome, ProgramVariable { name: Arc::from("y") }),
        ]);
        let program =
            FunctionalProgram::new(arena, schema, root, root, ProgramLimits::default()).unwrap();
        let program_wire = crate::functional_program_to_wire(&program).unwrap();
        let requirement = key_from_requirement(&program.factor_requirements()[0]);
        let query = antecedent_core::CausalQuery::Distribution(
            antecedent_core::InterventionalDistributionQuery::new(
                outcome,
                [antecedent_core::Intervention::set(treatment, Value::f64(1.0))],
            ),
        );
        let query_wire = crate::causal_query_to_wire(&query).unwrap();
        let result = InterventionalDistributionWire {
            atoms: vec![
                crate::DistributionAtomWire {
                    outcomes: vec![(1, ValueWire::Float64(0.0))],
                    conditioning: Vec::new(),
                    probability: 0.3,
                },
                crate::DistributionAtomWire {
                    outcomes: vec![(1, ValueWire::Float64(1.0))],
                    conditioning: Vec::new(),
                    probability: 0.7,
                },
            ],
        };
        let laws = DistributionFactorLawsWire {
            format: FACTOR_LAWS_FORMAT,
            provider: "empirical_table".into(),
            source_rows: 8,
            complete_case_rows: 8,
            missing_row_policy: "joint_complete_case".into(),
            domains: vec![
                (0, vec![ValueWire::Float64(0.0), ValueWire::Float64(1.0)]),
                (1, vec![ValueWire::Float64(0.0), ValueWire::Float64(1.0)]),
            ],
            requirements: vec![requirement.clone()],
            factors: vec![DistributionFactorTableWire {
                key: requirement,
                rows: vec![
                    DistributionFactorRowWire {
                        values: vec![ValueWire::Float64(0.0)],
                        probability: 0.3,
                    },
                    DistributionFactorRowWire {
                        values: vec![ValueWire::Float64(1.0)],
                        probability: 0.7,
                    },
                ],
            }],
        };
        (query_wire, result, program_wire, laws)
    }

    fn scalar_fixture() -> (CausalQueryWire, FunctionalProgramWire, DistributionFactorLawsWire) {
        use antecedent_core::{AverageEffectQuery, CausalQuery};
        use antecedent_expr::{ExprNode, OutcomeExprId};

        let treatment = VariableId::from_raw(0);
        let outcome = VariableId::from_raw(1);
        let mut arena = CausalExprArena::new();
        let outcome_set = arena.intern_var_set([outcome]);
        let empty = arena.empty_var_set();
        let no_intervention = arena.empty_intervention_set();
        let distribution = arena.intern_distribution(
            outcome_set,
            empty,
            no_intervention,
            DomainRef::Observational,
        );
        let root = arena.intern(ExprNode::Expectation {
            function: OutcomeExprId::identity(outcome),
            distribution,
        });
        let schema = ProgramSchema::new([
            (treatment, ProgramVariable { name: Arc::from("t") }),
            (outcome, ProgramVariable { name: Arc::from("y") }),
        ]);
        let program =
            FunctionalProgram::new(arena, schema, root, root, ProgramLimits::default()).unwrap();
        let key = key_from_requirement(&program.factor_requirements()[0]);
        let query = crate::causal_query_to_wire(&CausalQuery::AverageEffect(
            AverageEffectQuery::binary_ate(treatment, outcome),
        ))
        .unwrap();
        let laws = DistributionFactorLawsWire {
            format: FACTOR_LAWS_FORMAT,
            provider: "empirical_table".into(),
            source_rows: 4,
            complete_case_rows: 4,
            missing_row_policy: "joint_complete_case".into(),
            domains: vec![(1, vec![ValueWire::Float64(0.0), ValueWire::Float64(1.0)])],
            requirements: vec![key.clone()],
            factors: vec![DistributionFactorTableWire {
                key,
                rows: vec![
                    DistributionFactorRowWire {
                        values: vec![ValueWire::Float64(0.0)],
                        probability: 0.25,
                    },
                    DistributionFactorRowWire {
                        values: vec![ValueWire::Float64(1.0)],
                        probability: 0.75,
                    },
                ],
            }],
        };
        (query, crate::functional_program_to_wire(&program).unwrap(), laws)
    }

    #[test]
    fn replay_charges_quadratic_provider_and_atom_scans_before_work() {
        let (query, result, program, mut laws) = replay_fixture();
        let row = laws.factors[0].rows[0].clone();
        laws.factors[0].rows = vec![row; 128];
        let limits = ReplayLimits { max_operations: 1_000, ..ReplayLimits::default() };
        assert_eq!(
            super::replay_distribution_atoms(&query, &result, &program, &laws, &limits),
            Err(ContractReason::DistributionReplayResourceLimit)
        );

        let (query, mut result, program, laws) = replay_fixture();
        result.atoms = vec![result.atoms[0].clone(); 128];
        assert_eq!(
            super::replay_distribution_atoms(&query, &result, &program, &laws, &limits),
            Err(ContractReason::DistributionReplayResourceLimit)
        );
    }

    #[test]
    fn independent_replay_recomputes_distribution_atoms_from_factor_laws() {
        let (query, result, program, laws) = replay_fixture();
        replay_distribution_atoms(&query, &result, &program, &laws).unwrap();

        let mut changed_result = result.clone();
        changed_result.atoms[1].probability = 0.6;
        assert_eq!(
            replay_distribution_atoms(&query, &changed_result, &program, &laws),
            Err(ContractReason::DistributionAtomReplayMismatch)
        );

        let mut changed_laws = laws;
        changed_laws.factors[0].rows[0].probability = 0.2;
        changed_laws.factors[0].rows[1].probability = 0.8;
        assert_eq!(
            replay_distribution_atoms(&query, &result, &program, &changed_laws),
            Err(ContractReason::DistributionAtomReplayMismatch)
        );
    }

    #[test]
    fn functional_scalar_replay_recomputes_point_from_factor_laws() {
        let (query, program, laws) = scalar_fixture();
        replay_functional_scalar(&query, Some(0.75), &program, &laws).unwrap();
        assert_eq!(
            replay_functional_scalar(&query, Some(0.8), &program, &laws),
            Err(ContractReason::FunctionalEffectResultMismatch)
        );
    }

    #[test]
    fn functional_scalar_replay_refuses_missing_or_aliased_laws() {
        let (query, program, laws) = scalar_fixture();
        let mut missing = laws.clone();
        missing.factors.clear();
        assert_eq!(
            replay_functional_scalar(&query, Some(0.75), &program, &missing),
            Err(ContractReason::FunctionalEffectFactorLawCoverage)
        );

        let mut aliased = laws;
        aliased.factors[0].key.variables = vec![0];
        assert_eq!(
            replay_functional_scalar(&query, Some(0.75), &program, &aliased),
            Err(ContractReason::FunctionalEffectFactorLawCoverage)
        );
    }

    #[test]
    fn functional_scalar_replay_binds_query_and_program_root() {
        let (query, program, laws) = scalar_fixture();
        let wrong_query = crate::causal_query_to_wire(&antecedent_core::CausalQuery::Distribution(
            antecedent_core::InterventionalDistributionQuery::new(
                VariableId::from_raw(1),
                [antecedent_core::Intervention::set(VariableId::from_raw(0), Value::f64(1.0))],
            ),
        ))
        .unwrap();
        assert_eq!(
            replay_functional_scalar(&wrong_query, Some(0.75), &program, &laws),
            Err(ContractReason::FunctionalEffectQueryKind)
        );

        let mut changed_root = program;
        changed_root.executable = u32::MAX;
        assert_eq!(
            replay_functional_scalar(&query, Some(0.75), &changed_root, &laws),
            Err(ContractReason::FunctionalEffectFunctionalProgram)
        );
    }

    #[test]
    fn functional_scalar_replay_refuses_resource_exhaustion_and_zero_mass() {
        let (query, program, laws) = scalar_fixture();
        let mut too_large = laws.clone();
        too_large.domains[0].1 = (0..=MAX_REPLAY_CELLS)
            .map(|value| ValueWire::Int64(i64::try_from(value).expect("replay cell fits i64")))
            .collect();
        assert_eq!(
            replay_functional_scalar(&query, Some(0.75), &program, &too_large),
            Err(ContractReason::FunctionalEffectFactorLawsResourceLimit)
        );

        let mut zero_mass = laws;
        for row in &mut zero_mass.factors[0].rows {
            row.probability = 0.0;
        }
        assert_eq!(
            replay_functional_scalar(&query, Some(0.0), &program, &zero_mass),
            Err(ContractReason::FunctionalEffectFactorNormalization)
        );
    }

    #[test]
    fn caller_replay_limits_refuse_what_the_defaults_admit() {
        let (query, program, laws) = scalar_fixture();
        super::replay_functional_scalar(
            &query,
            Some(0.75),
            &program,
            &laws,
            &ReplayLimits::default(),
        )
        .unwrap();
        let few_cells = ReplayLimits { max_cells: 1, ..ReplayLimits::default() };
        assert_eq!(
            super::replay_functional_scalar(&query, Some(0.75), &program, &laws, &few_cells),
            Err(ContractReason::FunctionalEffectFactorLawsResourceLimit)
        );
        let few_operations = ReplayLimits { max_operations: 0, ..ReplayLimits::default() };
        assert_eq!(
            super::replay_functional_scalar(&query, Some(0.75), &program, &laws, &few_operations),
            Err(ContractReason::FunctionalEffectReplayResourceLimit)
        );
    }
}
