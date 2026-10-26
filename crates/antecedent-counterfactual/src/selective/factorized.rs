//! Exact variable elimination for finite SCMs whose joint exogenous space is too large to
//! enumerate.
//!
//! The enumerating route (`abduct_unit`) walks every joint state of the exogenous terms, so
//! its cost is the *product* of the term sizes. The quantity it computes,
//! `E[q(worlds) | evidence]`, depends on that joint law only through the factorization of
//! the structural equations, and this module evaluates it on that factorization instead.
//!
//! # Factor graph
//!
//! Per unit the graph holds
//!
//! * one variable per needed exogenous term `U_k` (domain: its states), with its pmf as a
//!   prior factor;
//! * one variable `X(w, i)` per needed (world, endogenous variable) pair, with the levels of
//!   the variable as its domain; and
//! * one indicator factor per needed non-intervened pair, over `X(w, i)`, the term `U_e(i)`
//!   and the parent variables the world reads (through its routes), equal to one exactly when
//!   the mechanism table maps those parent values and that term state to the level of
//!   `X(w, i)`.
//!
//! The factual world is one more world with no interventions and no routes. A hard
//! intervention makes `X(w, i)` a constant, and the factual evidence makes the observed factual
//! variables constants, so both shrink the tables instead of adding to them. Sharing one
//! `U_k` variable among every world and mechanism that reads it is exactly the coupling of the
//! enumerating route: a world cannot read a different state of a term than the factual world
//! did.
//!
//! # Elimination
//!
//! Every variable except the ones the quantity reads is summed out, smallest induced table
//! first (deterministic greedy order, ties broken by variable index). Each elimination
//! multiplies the factors that mention the variable and sums it out; the result is rescaled
//! to a unit maximum, with the scale accumulated in log space, so a long product of small
//! masses neither underflows nor loses the posterior ratios. The remaining factors form
//! `P(evidence, Q = q)` up to that scale, from which the evidence probability and the value are
//! read exactly as the enumerating route reads its posterior.
//!
//! The route is exact but not unconditionally cheap: its cost is exponential in the width of
//! the elimination order (the largest table it must form). A plan whose largest table exceeds
//! [`MAX_ELIMINATION_TABLE_ENTRIES`] is refused with
//! `cross_world_execution.elimination_bound_exceeded`; the model stays computable but not by
//! this bounded plan.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{BTreeMap, BTreeSet};

use antecedent_core::{CanonicalWriter, ExecutionContext};

use super::finite_scm::{CompiledScm, FactualUnit, FiniteQuantity, WorldPlan};
use super::refusal::{CrossWorldExecutionError, invalid, poll, refuse};

/// Most entries of any table the elimination forms (a factor or an intermediate product).
pub const MAX_ELIMINATION_TABLE_ENTRIES: usize = 1 << 22;
/// Most variables (exogenous terms plus world variables) of one unit's factor graph.
pub const MAX_ELIMINATION_NODES: usize = 1 << 13;
/// Most multiply-accumulate operations of one request, summed over its units.
pub const MAX_ELIMINATION_WORK: u64 = 1 << 32;

const POLL_EVERY_CELLS: usize = 1 << 15;

/// What one unit's elimination did.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct UnitInference {
    /// The quantity for the unit.
    pub value: f64,
    /// `P(evidence)` under the mechanisms.
    pub evidence_probability: f64,
    /// Variables of the factor graph.
    pub nodes: u64,
    /// Factors the graph started with.
    pub factors: u64,
    /// Entries of the largest table formed.
    pub peak_table_entries: u64,
    /// Multiply-accumulate operations.
    pub multiplications: u64,
    /// Digest of the elimination order and the largest scope it formed.
    pub plan_digest: String,
}

/// A table over the variables `vars` (ascending node index), row-major with the first variable
/// most significant.
struct Factor {
    vars: Vec<usize>,
    data: Vec<f64>,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Label {
    Exogenous(usize),
    World { world: usize, variable: usize },
}

fn bound_exceeded(what: &str) -> CrossWorldExecutionError {
    refuse(
        antecedent_core::reason_code!("route_not_supported"),
        "cross_world_execution.elimination_bound_exceeded",
        Some(what.to_owned()),
        "the exact elimination order needs a table above the documented bound for this \
         declaration, evidence and quantity; reduce the levels, parents or exogenous states, or \
         query fewer coupled worlds. The model is declared, so the counterfactual is computable \
         but not by this bounded plan",
    )
}

fn outside_support(unit: &str) -> CrossWorldExecutionError {
    refuse(
        antecedent_core::reason_code!("route_not_supported"),
        "cross_world_execution.evidence_outside_model_support",
        Some(unit.to_owned()),
        "the factual evidence has probability zero under the declared mechanisms; revise \
         the mechanisms or the evidence (a refuted model is not repaired by conditioning)",
    )
}

fn product(dims: impl Iterator<Item = usize>) -> usize {
    dims.fold(1usize, usize::saturating_mul)
}

/// The elimination work accumulated over a request's units.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct EliminationWork {
    /// Multiply-accumulate operations so far.
    pub multiplications: u64,
}

/// Exact `E[q | evidence]` for one unit by variable elimination.
///
/// `plans` holds one plan per request world followed by the factual world's plan (no
/// interventions, no routes); the factual world is `plans.len() - 1`.
///
/// # Errors
/// Cancellation; evidence with zero probability (`evidence_outside_model_support`) or a positive
/// probability below the smallest `f64` (`evidence_probability_underflow`); a plan above the
/// table or node bound (`elimination_bound_exceeded`); a request above the work budget
/// (`bounds_exceeded`).
#[allow(
    clippy::too_many_lines,
    reason = "one pass: build the factor graph, then eliminate in a fixed greedy order"
)]
pub(crate) fn infer_unit(
    compiled: &CompiledScm<'_>,
    plans: &[WorldPlan],
    quantity: &FiniteQuantity,
    unit: &FactualUnit,
    work: &mut EliminationWork,
    ctx: &ExecutionContext,
) -> Result<UnitInference, CrossWorldExecutionError> {
    let scm = compiled.scm;
    let n_vars = scm.variables.len();
    let factual = plans.len() - 1;

    // ---- which (world, variable) pairs and exogenous terms the answer depends on ----
    let mut needed = vec![vec![false; n_vars]; plans.len()];
    for (world, variable) in quantity.atoms() {
        needed[world][variable] = true;
    }
    for (variable, _) in &unit.observed {
        needed[factual][*variable] = true;
    }
    let mut exogenous_needed = vec![false; scm.exogenous.len()];
    // Parents precede children, so a descending sweep sees every dependent before its inputs.
    for variable in (0..n_vars).rev() {
        for (world, plan) in plans.iter().enumerate() {
            if !needed[world][variable] || plan.fixed[variable].is_some() {
                continue;
            }
            let var = &scm.variables[variable];
            exogenous_needed[var.exogenous] = true;
            for (slot, parent) in var.parents.iter().enumerate() {
                needed[plan.parent_source[variable][slot]][*parent] = true;
            }
        }
    }
    let observed: BTreeMap<usize, u8> = unit.observed.iter().copied().collect();
    let constant = |world: usize, variable: usize| -> Option<u8> {
        plans[world].fixed[variable]
            .or_else(|| if world == factual { observed.get(&variable).copied() } else { None })
    };

    // ---- variables ----
    let mut labels: Vec<Label> = Vec::new();
    let mut domains: Vec<usize> = Vec::new();
    let mut exogenous_node: Vec<Option<usize>> = vec![None; scm.exogenous.len()];
    for (k, exo) in scm.exogenous.iter().enumerate() {
        if exogenous_needed[k] {
            exogenous_node[k] = Some(labels.len());
            labels.push(Label::Exogenous(k));
            domains.push(exo.pmf.len());
        }
    }
    let mut world_node: BTreeMap<(usize, usize), usize> = BTreeMap::new();
    for variable in 0..n_vars {
        for world in 0..plans.len() {
            if needed[world][variable] && constant(world, variable).is_none() {
                world_node.insert((world, variable), labels.len());
                labels.push(Label::World { world, variable });
                domains.push(scm.variables[variable].levels);
            }
        }
    }
    if labels.len() > MAX_ELIMINATION_NODES {
        return Err(bound_exceeded("variables"));
    }

    // ---- factors ----
    let mut factors: Vec<Option<Factor>> = Vec::new();
    let mut cells_since_poll = 0usize;
    for (k, node) in exogenous_node.iter().enumerate() {
        if let Some(node) = node {
            factors.push(Some(Factor { vars: vec![*node], data: compiled.pmfs[k].clone() }));
        }
    }
    for variable in 0..n_vars {
        let var = &scm.variables[variable];
        for world in 0..plans.len() {
            if !needed[world][variable] || plans[world].fixed[variable].is_some() {
                continue;
            }
            let own = constant(world, variable);
            // Roles within the factor: the output variable (unless observed), the exogenous
            // term, and each parent (unless it is a constant in the world it is read from).
            let out_node = world_node.get(&(world, variable)).copied();
            let u_node = exogenous_node[var.exogenous]
                .ok_or_else(|| invalid("cross_world_execution.invalid_scm", var.name.clone()))?;
            let parent_nodes: Vec<(Option<usize>, Option<u8>)> = var
                .parents
                .iter()
                .enumerate()
                .map(|(slot, parent)| {
                    let source = plans[world].parent_source[variable][slot];
                    (world_node.get(&(source, *parent)).copied(), constant(source, *parent))
                })
                .collect();
            let mut scope: Vec<usize> =
                out_node.into_iter().chain([u_node]).chain(parent_nodes.iter().filter_map(|p| p.0)).collect();
            scope.sort_unstable();
            scope.dedup();
            let dims: Vec<usize> = scope.iter().map(|v| domains[*v]).collect();
            let size = product(dims.iter().copied());
            if size > MAX_ELIMINATION_TABLE_ENTRIES {
                return Err(bound_exceeded(&var.name));
            }
            let position = |node: usize| scope.iter().position(|v| *v == node);
            let u_at = position(u_node);
            let out_at = out_node.and_then(position);
            let parent_at: Vec<Option<usize>> =
                parent_nodes.iter().map(|p| p.0.and_then(position)).collect();
            let mut assignment = vec![0usize; scope.len()];
            let mut data = Vec::with_capacity(size);
            for _ in 0..size {
                cells_since_poll += 1;
                if cells_since_poll >= POLL_EVERY_CELLS {
                    cells_since_poll = 0;
                    poll(ctx)?;
                }
                let u_state = u_at.map_or(0, |at| assignment[at]);
                let mut cfg = 0usize;
                for (slot, parent) in var.parents.iter().enumerate() {
                    let level = match (parent_at[slot], parent_nodes[slot].1) {
                        (Some(at), _) => assignment[at],
                        (None, Some(fixed)) => usize::from(fixed),
                        (None, None) => 0,
                    };
                    cfg = cfg * scm.variables[*parent].levels + level;
                }
                let produced = usize::from(compiled.mechanism(variable, u_state, cfg));
                let target = match (out_at, own) {
                    (Some(at), _) => assignment[at],
                    (None, Some(fixed)) => usize::from(fixed),
                    (None, None) => 0,
                };
                data.push(if produced == target { 1.0 } else { 0.0 });
                // Odometer, last variable fastest.
                for at in (0..assignment.len()).rev() {
                    assignment[at] += 1;
                    if assignment[at] < dims[at] {
                        break;
                    }
                    assignment[at] = 0;
                }
            }
            factors.push(Some(Factor { vars: scope, data }));
        }
    }
    let n_factors = factors.len();

    // ---- query variables (the world variables the quantity reads) ----
    let query: BTreeSet<usize> = quantity
        .atoms()
        .into_iter()
        .filter_map(|(world, variable)| world_node.get(&(world, variable)).copied())
        .collect();

    // ---- interaction graph and greedy elimination ----
    let mut holders: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); labels.len()];
    let mut adjacent: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); labels.len()];
    for (id, factor) in factors.iter().enumerate() {
        let Some(factor) = factor else { continue };
        for &a in &factor.vars {
            holders[a].insert(id);
            for &b in &factor.vars {
                if a != b {
                    adjacent[a].insert(b);
                }
            }
        }
    }
    let weight_of = |node: usize, adjacent: &[BTreeSet<usize>]| -> u64 {
        let cells = product(adjacent[node].iter().map(|v| domains[*v]));
        u64::try_from(cells).unwrap_or(u64::MAX)
    };
    let mut queue: BTreeSet<(u64, usize)> = BTreeSet::new();
    let mut weight: Vec<u64> = vec![0; labels.len()];
    for node in (0..labels.len()).filter(|n| !query.contains(n)) {
        weight[node] = weight_of(node, &adjacent);
        queue.insert((weight[node], node));
    }

    let mut log_scale = 0.0f64;
    let mut peak = 0usize;
    let mut multiplications = 0u64;
    let mut order = CanonicalWriter::new("antecedent.counterfactual.selective.elimination_order.v1");
    order.word(labels.len() as u64);
    while let Some((cells, x)) = queue.pop_first() {
        poll(ctx)?;
        if usize::try_from(cells).map_or(true, |c| c > MAX_ELIMINATION_TABLE_ENTRIES) {
            return Err(bound_exceeded("elimination_table"));
        }
        match labels[x] {
            Label::Exogenous(k) => order.word(0).word(k as u64),
            Label::World { world, variable } => order.word(1).word(world as u64).word(variable as u64),
        };
        let ids: Vec<usize> = holders[x].iter().copied().collect();
        let mut inputs: Vec<Factor> = Vec::with_capacity(ids.len());
        for id in &ids {
            if let Some(factor) = factors[*id].take() {
                for v in &factor.vars {
                    holders[*v].remove(id);
                }
                inputs.push(factor);
            }
        }
        let outcome = combine(&inputs, Some(x), &domains, ctx)?;
        multiplications = multiplications.saturating_add(outcome.multiplications);
        work.multiplications = work.multiplications.saturating_add(outcome.multiplications);
        if work.multiplications > MAX_ELIMINATION_WORK {
            return Err(invalid("cross_world_execution.bounds_exceeded", "elimination_work"));
        }
        peak = peak.max(outcome.factor.data.len());
        let Some(scaled) = rescale(outcome.factor, &mut log_scale) else {
            return Err(outside_support(&unit.unit));
        };
        // Fill-in: every neighbour of `x` now interacts with every other neighbour.
        let neighbours: Vec<usize> = adjacent[x].iter().copied().collect();
        for &a in &neighbours {
            adjacent[a].remove(&x);
            for &b in &neighbours {
                if a != b {
                    adjacent[a].insert(b);
                }
            }
        }
        adjacent[x].clear();
        let new_id = factors.len();
        for &v in &scaled.vars {
            holders[v].insert(new_id);
        }
        factors.push(Some(scaled));
        for &a in &neighbours {
            if !query.contains(&a) {
                queue.remove(&(weight[a], a));
                weight[a] = weight_of(a, &adjacent);
                queue.insert((weight[a], a));
            }
        }
    }

    // ---- the answer table over the query variables ----
    let remaining: Vec<Factor> = factors.into_iter().flatten().collect();
    let joint = combine(&remaining, None, &domains, ctx)?;
    multiplications = multiplications.saturating_add(joint.multiplications);
    work.multiplications = work.multiplications.saturating_add(joint.multiplications);
    if work.multiplications > MAX_ELIMINATION_WORK {
        return Err(invalid("cross_world_execution.bounds_exceeded", "elimination_work"));
    }
    peak = peak.max(joint.factor.data.len());
    let Some(table) = rescale(joint.factor, &mut log_scale) else {
        return Err(outside_support(&unit.unit));
    };
    let total: f64 = table.data.iter().sum();
    if !(total.is_finite() && total > 0.0) {
        return Err(outside_support(&unit.unit));
    }
    let evidence_probability = (log_scale + total.ln()).exp();
    if evidence_probability == 0.0 {
        return Err(refuse(
            antecedent_core::reason_code!("route_not_supported"),
            "cross_world_execution.evidence_probability_underflow",
            Some(unit.unit.clone()),
            "use an executor with probability arithmetic that can retain this positive evidence mass; this f64 route cannot represent it",
        ));
    }

    // The quantity over the answer table, reading constants for intervened atoms.
    let mut value = 0.0;
    let dims: Vec<usize> = table.vars.iter().map(|v| domains[*v]).collect();
    let mut assignment = vec![0usize; table.vars.len()];
    for mass in &table.data {
        let at = |world: usize, variable: usize| -> u8 {
            match world_node.get(&(world, variable)) {
                Some(node) => {
                    let position = table.vars.iter().position(|v| v == node);
                    // Levels fit the `u8` level code by the declaration bound.
                    position.map_or(0, |p| u8::try_from(assignment[p]).unwrap_or(u8::MAX))
                }
                None => constant(world, variable).unwrap_or(0),
            }
        };
        value += (mass / total) * quantity.at_lookup(at);
        for p in (0..assignment.len()).rev() {
            assignment[p] += 1;
            if assignment[p] < dims[p] {
                break;
            }
            assignment[p] = 0;
        }
    }

    let mut plan = CanonicalWriter::new("antecedent.counterfactual.selective.elimination_plan.v1");
    plan.text(&order.finish_hex()).word(peak as u64).word(n_factors as u64);
    Ok(UnitInference {
        value,
        evidence_probability,
        nodes: labels.len() as u64,
        factors: n_factors as u64,
        peak_table_entries: peak as u64,
        multiplications,
        plan_digest: plan.finish_hex(),
    })
}

/// The product of `inputs` over the union of their variables, with `sum_out` summed away.
struct Combined {
    factor: Factor,
    multiplications: u64,
}

fn combine(
    inputs: &[Factor],
    sum_out: Option<usize>,
    domains: &[usize],
    ctx: &ExecutionContext,
) -> Result<Combined, CrossWorldExecutionError> {
    let mut union: BTreeSet<usize> = BTreeSet::new();
    for factor in inputs {
        union.extend(factor.vars.iter().copied());
    }
    let kept: Vec<usize> = union.iter().copied().filter(|v| Some(*v) != sum_out).collect();
    let kept_dims: Vec<usize> = kept.iter().map(|v| domains[*v]).collect();
    let out_size = product(kept_dims.iter().copied());
    if out_size > MAX_ELIMINATION_TABLE_ENTRIES {
        return Err(bound_exceeded("elimination_table"));
    }
    let summed = sum_out.map_or(1, |x| domains[x]);
    // Strides of each input factor along each kept variable and along the summed variable.
    let strides_of = |factor: &Factor| -> Vec<usize> {
        let mut strides = vec![0usize; factor.vars.len()];
        let mut acc = 1usize;
        for at in (0..factor.vars.len()).rev() {
            strides[at] = acc;
            acc = acc.saturating_mul(domains[factor.vars[at]]);
        }
        strides
    };
    let all_strides: Vec<Vec<usize>> = inputs.iter().map(strides_of).collect();
    let kept_stride: Vec<Vec<usize>> = inputs
        .iter()
        .zip(&all_strides)
        .map(|(factor, strides)| {
            kept.iter()
                .map(|v| factor.vars.iter().position(|w| w == v).map_or(0, |at| strides[at]))
                .collect()
        })
        .collect();
    let summed_stride: Vec<usize> = inputs
        .iter()
        .zip(&all_strides)
        .map(|(factor, strides)| {
            sum_out
                .and_then(|x| factor.vars.iter().position(|w| *w == x))
                .map_or(0, |at| strides[at])
        })
        .collect();

    let mut data = Vec::with_capacity(out_size);
    let mut assignment = vec![0usize; kept.len()];
    let mut bases = vec![0usize; inputs.len()];
    let mut cells = 0usize;
    for _ in 0..out_size {
        cells += 1;
        if cells >= POLL_EVERY_CELLS / summed.max(1) + 1 {
            cells = 0;
            poll(ctx)?;
        }
        let mut sum = 0.0;
        for level in 0..summed {
            let mut term = 1.0;
            for (f, factor) in inputs.iter().enumerate() {
                term *= factor.data[bases[f] + level * summed_stride[f]];
                if term == 0.0 {
                    break;
                }
            }
            sum += term;
        }
        data.push(sum);
        for at in (0..assignment.len()).rev() {
            assignment[at] += 1;
            for (f, base) in bases.iter_mut().enumerate() {
                *base += kept_stride[f][at];
            }
            if assignment[at] < kept_dims[at] {
                break;
            }
            for (f, base) in bases.iter_mut().enumerate() {
                *base -= kept_stride[f][at] * kept_dims[at];
            }
            assignment[at] = 0;
        }
    }
    let multiplications = (out_size as u64)
        .saturating_mul(summed as u64)
        .saturating_mul(inputs.len().max(1) as u64);
    Ok(Combined { factor: Factor { vars: kept, data }, multiplications })
}

/// Divide a factor by its largest entry, adding the logarithm of that entry to `log_scale`;
/// `None` when every entry is zero.
fn rescale(mut factor: Factor, log_scale: &mut f64) -> Option<Factor> {
    let max = factor.data.iter().copied().fold(0.0f64, f64::max);
    if !(max.is_finite() && max > 0.0) {
        return None;
    }
    if max != 1.0 {
        for entry in &mut factor.data {
            *entry /= max;
        }
        *log_scale += max.ln();
    }
    Some(factor)
}
