//! Canonical identity text of a compiled product.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

pub(super) fn write_estimands(writer: &mut CanonicalWriter, estimands: &[IdentifiedEstimand]) {
    writer.word(count(estimands.len()));
    for estimand in estimands {
        writer.text(&estimand.method);
        write_vars(writer, &estimand.adjustment_set);
        write_vars(writer, &estimand.instruments);
        write_vars(writer, &estimand.mediators);
        writer.word(u64::from(estimand.functional.raw()));
        match estimand.rd_design {
            None => {
                writer.word(0);
            }
            Some(design) => {
                writer.word(1);
                writer.word(u64::from(design.running_variable.raw()));
                writer.word(design.cutoff.to_bits());
                writer.word(design.bandwidth.to_bits());
            }
        }
    }
}

pub(super) fn write_vars(writer: &mut CanonicalWriter, variables: &[VariableId]) {
    writer.word(count(variables.len()));
    for variable in variables {
        writer.word(u64::from(variable.raw()));
    }
}

pub(super) fn write_arena(writer: &mut CanonicalWriter, arena: &CausalExprArena) {
    writer.word(count(arena.len()));
    for index in 0..arena.len() {
        write_node(writer, arena.node(ExprId::from_raw(nth_raw(index))));
    }
    writer.word(count(arena.var_set_count()));
    for index in 0..arena.var_set_count() {
        write_vars(writer, arena.var_set(VarSetId::from_raw(nth_raw(index))));
    }
    writer.word(count(arena.intervention_set_count()));
    for index in 0..arena.intervention_set_count() {
        let assignments =
            arena.intervention_assignments(InterventionSetId::from_raw(nth_raw(index)));
        writer.word(count(assignments.len()));
        for assignment in assignments {
            writer.word(u64::from(assignment.variable.raw()));
            assignment.value.encode(writer);
        }
    }
    writer.word(count(arena.list_count()));
    for index in 0..arena.list_count() {
        let list = arena.list(ExprListId::from_raw(nth_raw(index)));
        writer.word(count(list.len()));
        for expr in list {
            writer.word(u64::from(expr.raw()));
        }
    }
    writer.word(count(arena.population_count()));
    for index in 0..arena.population_count() {
        writer.text(arena.population(PopulationKeyId::from_raw(nth_raw(index))));
    }
}

pub(super) fn write_node(writer: &mut CanonicalWriter, node: &ExprNode) {
    match node {
        ExprNode::Distribution {
            variables,
            conditioned_on,
            intervention,
            domain,
            population,
            regime,
        } => {
            writer.word(1);
            writer.word(u64::from(variables.raw()));
            writer.word(u64::from(conditioned_on.raw()));
            writer.word(u64::from(intervention.raw()));
            write_domain(writer, *domain);
            writer.word(u64::from(population.raw()));
            writer.opt_word(regime.map(|id| u64::from(id.raw())));
        }
        ExprNode::Kernel { body, bound, population, regime } => {
            writer.word(2);
            writer.word(u64::from(body.raw()));
            writer.word(u64::from(bound.raw()));
            writer.word(u64::from(population.raw()));
            writer.opt_word(regime.map(|id| u64::from(id.raw())));
        }
        ExprNode::Product(list) => {
            writer.word(3);
            writer.word(u64::from(list.raw()));
        }
        ExprNode::SumOut { variables, expr } => {
            writer.word(4);
            writer.word(u64::from(variables.raw()));
            writer.word(u64::from(expr.raw()));
        }
        ExprNode::IntegralOut { variables, expr } => {
            writer.word(5);
            writer.word(u64::from(variables.raw()));
            writer.word(u64::from(expr.raw()));
        }
        ExprNode::Ratio { numerator, denominator } => {
            writer.word(6);
            writer.word(u64::from(numerator.raw()));
            writer.word(u64::from(denominator.raw()));
        }
        ExprNode::Expectation { function, distribution } => {
            writer.word(7);
            writer.word(u64::from(function.variable().raw()));
            writer.word(u64::from(distribution.raw()));
        }
        ExprNode::Contrast { left, right, op } => {
            writer.word(8);
            writer.word(u64::from(left.raw()));
            writer.word(u64::from(right.raw()));
            write_contrast_op(writer, *op);
        }
    }
}

pub(super) fn write_domain(writer: &mut CanonicalWriter, domain: DomainRef) {
    match domain {
        DomainRef::Observational => {
            writer.word(1);
        }
        DomainRef::Interventional => {
            writer.word(2);
        }
    }
}

pub(super) fn write_contrast_op(writer: &mut CanonicalWriter, op: ContrastOp) {
    match op {
        ContrastOp::Difference => {
            writer.word(1);
        }
    }
}

pub(super) fn write_derivation(writer: &mut CanonicalWriter, derivation: &crate::result::DerivationTrace) {
    writer.word(count(derivation.steps.len()));
    for step in &derivation.steps {
        writer.text(&step.rule);
        writer.text(&step.detail);
    }
}

pub(super) fn write_assumptions(writer: &mut CanonicalWriter, assumptions: &AssumptionSet) {
    writer.word(count(assumptions.len()));
    for record in &assumptions.entries {
        write_assumption_record(writer, record);
    }
}

pub(super) fn write_assumption_record(writer: &mut CanonicalWriter, record: &AssumptionRecord) {
    record.assumption.encode(writer);
    write_assumption_source(writer, &record.source);
    write_assumption_scope(writer, &record.scope);
    writer.word(assumption_status_code(record.status));
}

pub(super) fn write_assumption_source(writer: &mut CanonicalWriter, source: &AssumptionSource) {
    match source {
        AssumptionSource::UserDeclared => {
            writer.word(1);
        }
        AssumptionSource::AlgorithmDefault { algorithm } => {
            writer.word(2);
            writer.text(algorithm);
        }
        AssumptionSource::Artifact => {
            writer.word(3);
        }
        AssumptionSource::Derived { from } => {
            writer.word(4);
            writer.text(from);
        }
    }
}

pub(super) fn write_assumption_scope(writer: &mut CanonicalWriter, scope: &AssumptionScope) {
    match scope {
        AssumptionScope::Global => {
            writer.word(1);
        }
        AssumptionScope::Identification => {
            writer.word(2);
        }
        AssumptionScope::Estimation => {
            writer.word(3);
        }
        AssumptionScope::Discovery => {
            writer.word(4);
        }
        AssumptionScope::Variables { variables } => {
            writer.word(5);
            write_vars(writer, variables);
        }
    }
}

pub(super) const fn assumption_status_code(status: AssumptionStatus) -> u64 {
    match status {
        AssumptionStatus::Declared => 1,
        AssumptionStatus::Supported => 2,
        AssumptionStatus::Contradicted => 3,
        AssumptionStatus::Untestable => 4,
    }
}

pub(super) fn write_factors(writer: &mut CanonicalWriter, factors: &[RequiredFactor]) {
    writer.word(count(factors.len()));
    for factor in factors {
        writer.text(&factor.name);
        write_vars(writer, &factor.variables);
    }
}

pub(super) fn write_obligations(writer: &mut CanonicalWriter, obligations: &[UnresolvedObligation]) {
    writer.word(count(obligations.len()));
    for obligation in obligations {
        writer.text(&obligation.code);
        writer.text(&obligation.detail);
    }
}

pub(super) fn write_worlds(writer: &mut CanonicalWriter, worlds: &[IdentificationObligation]) {
    writer.word(count(worlds.len()));
    for world in worlds {
        let raw = world.world.map_or(0, |id| u64::from(id.raw()));
        writer.word(raw);
        writer.word(outcome_code(world.outcome));
        writer.word(claim_code(world.claim_class));
        write_obligations(writer, &world.unresolved);
    }
}

pub(super) fn write_assembled(writer: &mut CanonicalWriter, assembled: Option<&IdentificationObligation>) {
    match assembled {
        None => {
            writer.word(0);
        }
        Some(obligation) => {
            writer.word(1);
            writer.word(outcome_code(obligation.outcome));
            writer.word(claim_code(obligation.claim_class));
            write_obligations(writer, &obligation.unresolved);
        }
    }
}
