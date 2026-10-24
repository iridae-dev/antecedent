//! Shared fixtures of the path-specific tests that compare an identified functional with the
//! effect an enumerated latent-variable structural model produces.
//!
//! The reference solves the model with the treatment read at one level along the selected
//! edges and at the control level along the others, sharing every exogenous state between
//! the worlds; it uses no identification code.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::needless_range_loop,
    dead_code,
    reason = "test fixtures: small exact indices, levels and enumerated probabilities"
)]

#[path = "latent_scm.rs"]
pub mod latent_scm;

use std::sync::Arc;

use antecedent_core::{CausalQuery, Value, VariableId};
use antecedent_expr::{Assignment, DistributionProvider, EvalContext, EvalError, FactorSpec};
use antecedent_graph::{Admg, DenseNodeId};
use antecedent_identify::{IdentificationResult, IdentificationWorkspace, PathSpecificIdentifier};
use latent_scm::LatentScm;

pub fn v(i: usize) -> VariableId {
    VariableId::from_raw(i as u32)
}

pub fn d(i: usize) -> DenseNodeId {
    DenseNodeId::from_raw(i as u32)
}

/// Observational joint of a model with the first variable slowest, as `LatentScm::flat`.
pub struct JointProvider {
    pub cards: Vec<usize>,
    pub joint: Vec<f64>,
}

impl JointProvider {
    fn level_of(&self, cell: usize, variable: usize) -> usize {
        let stride: usize = self.cards[variable + 1..].iter().product();
        (cell / stride) % self.cards[variable]
    }

    fn mass(&self, constraints: &[(usize, usize)]) -> f64 {
        self.joint
            .iter()
            .enumerate()
            .filter(|(cell, _)| {
                constraints.iter().all(|&(var, level)| self.level_of(*cell, var) == level)
            })
            .map(|(_, p)| *p)
            .sum()
    }

    fn constraints(
        vars: &[VariableId],
        assignment: &Assignment,
    ) -> Result<Vec<(usize, usize)>, EvalError> {
        vars.iter()
            .map(|var| {
                let value = assignment
                    .get(*var)
                    .and_then(Value::as_f64)
                    .ok_or(EvalError::MissingBinding(*var))?;
                Ok((var.raw() as usize, value as usize))
            })
            .collect()
    }
}

impl DistributionProvider for JointProvider {
    fn probability(
        &self,
        spec: &FactorSpec<'_>,
        assignment: &Assignment,
        _: &EvalContext,
    ) -> Result<f64, EvalError> {
        let mut constraints = Self::constraints(spec.conditioned_on, assignment)?;
        let denominator = self.mass(&constraints);
        constraints.extend(Self::constraints(spec.variables, assignment)?);
        if denominator == 0.0 {
            return Err(EvalError::DivisionByZero);
        }
        Ok(self.mass(&constraints) / denominator)
    }

    fn support(
        &self,
        vars: &[VariableId],
        _: &EvalContext,
    ) -> Result<Arc<[Arc<[Value]>]>, EvalError> {
        let cards: Vec<usize> = vars.iter().map(|var| self.cards[var.raw() as usize]).collect();
        let rows: usize = cards.iter().product();
        Ok((0..rows)
            .map(|row| {
                let mut rest = row;
                let mut levels = vec![0usize; cards.len()];
                for i in (0..cards.len()).rev() {
                    levels[i] = rest % cards[i];
                    rest /= cards[i];
                }
                levels.into_iter().map(|l| Value::f64(l as f64)).collect()
            })
            .collect())
    }

    fn outcome(
        &self,
        var: VariableId,
        assignment: &Assignment,
        _: &EvalContext,
    ) -> Result<f64, EvalError> {
        assignment.get(var).and_then(Value::as_f64).ok_or(EvalError::MissingBinding(var))
    }

    fn n_draws(&self) -> Option<usize> {
        None
    }
}

/// Solve one exogenous state with the treatment fixed at `t_value` for its own node and read
/// at `edge_level(child)` by each child's equation.
pub fn solve_with_edge_levels(
    scm: &LatentScm,
    latent: &[usize],
    types: &[usize],
    treatment: usize,
    t_value: usize,
    edge_level: &dyn Fn(usize) -> usize,
) -> Vec<usize> {
    let mut values = vec![0usize; scm.cards.len()];
    for node in scm.topo() {
        if node == treatment {
            values[node] = t_value;
            continue;
        }
        let mut index = 0usize;
        for &p in &scm.parents[node] {
            let seen = if p == treatment { edge_level(node) } else { values[p] };
            index = index * scm.cards[p] + seen;
        }
        for k in scm.latents_of(node) {
            index = index * 2 + latent[k];
        }
        values[node] = scm.types[node][types[node]].table[index];
    }
    values
}

/// `E[Y with the edges into `selected` read at 1 and every other edge at 0] - E[Y with all
/// edges at 0]`, sharing the exogenous state between the two worlds.
pub fn path_specific_truth(
    scm: &LatentScm,
    treatment: usize,
    outcome: usize,
    selected: &[usize],
) -> f64 {
    let mut total = 0.0;
    for (p, latent, types) in scm.states() {
        let routed = solve_with_edge_levels(scm, &latent, &types, treatment, 1, &|child| {
            usize::from(selected.contains(&child))
        });
        let reference = solve_with_edge_levels(scm, &latent, &types, treatment, 0, &|_| 0);
        total += p * (routed[outcome] as f64 - reference[outcome] as f64);
    }
    total
}

pub fn admg(n: usize, directed: &[(usize, usize)], bidirected: &[(usize, usize)]) -> Admg {
    let mut graph = Admg::with_variables(n as u32);
    for &(a, b) in directed {
        graph.insert_directed(d(a), d(b)).unwrap();
    }
    for &(a, b) in bidirected {
        graph.insert_bidirected(d(a), d(b)).unwrap();
    }
    graph
}

pub fn identify(graph: &Admg, query: &CausalQuery) -> IdentificationResult {
    let id = PathSpecificIdentifier::new();
    let prepared = id.prepare(graph).unwrap();
    let mut workspace = IdentificationWorkspace::default();
    id.identify(&prepared, query, &mut workspace).unwrap()
}

pub fn evaluate(result: &IdentificationResult, scm: &LatentScm) -> f64 {
    let provider = JointProvider { cards: scm.cards.clone(), joint: scm.observational() };
    let functional = result.estimands[0].functional;
    assert!(
        result.arena.free_variables(functional).is_empty(),
        "the path-specific functional must be closed: {}",
        result.arena.pretty(functional)
    );
    let plan = result.arena.compile(functional).unwrap();
    plan.evaluate_with(&result.arena, &provider, &EvalContext::default(), &Assignment::new())
        .unwrap()
}

pub fn has_rule(result: &IdentificationResult, rule: &str) -> bool {
    result.derivation.steps.iter().any(|step| &*step.rule == rule)
}

