// SPDX-License-Identifier: MIT OR Apache-2.0
//! Numerical finite-law evaluation of retained cyclic functional programs.
//! Scientific identification and source authorization belong to the caller.
use crate::error::EstimationError;
use antecedent_core::{ExecutionContext, Value, VariableId};
use antecedent_expr::{Assignment, DistributionProvider, EvalContext, EvalError, FactorSpec};
use serde::{Deserialize, Serialize};
use std::{cell::Cell, sync::Arc};

/// Complete joint observational law on a declared finite product space.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FiniteCyclicLaw {
    /// Finite numeric values for each dense variable, with no duplicates.
    pub support: Vec<Vec<f64>>,
    /// Nonnegative joint weights, variable zero varying fastest.
    /// Integer counts give an empirical plug-in distribution, without a nominal interval claim.
    pub weights: Vec<f64>,
}
/// A bound assignment and its evaluated probability/density functional.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CyclicFunctionalValue {
    /// Dense variable-value bindings retained in the request order.
    pub assignment: Vec<(u32, f64)>,
    /// Evaluated value; absent when support/positivity conditions fail.
    pub value: Option<f64>,
    /// Located evaluator failure; no failed value is renormalized away.
    pub failure: Option<String>,
}
/// Identified functional evaluated conditional on its declared observational law.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CyclicFunctionalReport {
    /// Identity of the independently replayable graph identification source.
    pub source_identity: String,
    /// Actual theorem assumptions and derivation rechecked on consumption.
    pub derivation: Vec<String>,
    /// Retained assignments, including explicit positivity failures.
    pub values: Vec<CyclicFunctionalValue>,
    /// Actual bounded provider operations.
    pub operations: u64,
}
/// Borrowed numerical inputs from a retained functional program.
///
/// This component does not identify a causal effect or validate the provenance of
/// the supplied labels. Workflow callers must independently establish the program's
/// scientific authority before invoking numerical evaluation.
#[derive(Clone, Copy, Debug)]
pub struct CyclicFunctionalInputs<'a> {
    /// Number of dense observed variables.
    pub nodes: usize,
    /// Whether the identified program uses continuous semantics.
    pub continuous: bool,
    /// Independently established derivation, retained verbatim in the numerical report.
    pub derivation: &'a [String],
    /// Independently established source identity; not authenticated by this component.
    pub identity: &'a str,
    /// Whether the program came from an observational calculus derivation.
    pub calculus: bool,
    /// Retained executable expression; this component grants no identification claim.
    pub program: &'a antecedent_expr::FunctionalProgram,
}
fn invalid(s: impl std::fmt::Display) -> EstimationError {
    EstimationError::data_msg(s.to_string())
}
struct Provider<'a> {
    law: &'a FiniteCyclicLaw,
    strides: Vec<usize>,
    steps: Cell<u64>,
    ctx: &'a ExecutionContext,
}
impl Provider<'_> {
    fn tick(&self, count: usize) -> Result<(), EvalError> {
        self.steps.set(self.steps.get().saturating_add(count as u64));
        if self.ctx.cancellation.is_cancelled() {
            return Err(EvalError::ProviderKind("cancelled"));
        }
        if self.steps.get() > 50_000_000 {
            return Err(EvalError::ProviderKind("finite functional operation budget exhausted"));
        }
        Ok(())
    }
    #[allow(
        clippy::float_cmp,
        reason = "finite support values are exact category identities, not numerical proximity tests"
    )]
    fn mass(&self, variables: &[VariableId], assignment: &Assignment) -> Result<f64, EvalError> {
        let constraints: Vec<_> = variables
            .iter()
            .map(|&v| {
                let i = v.raw() as usize;
                let value = assignment
                    .get(v)
                    .and_then(Value::as_f64)
                    .ok_or(EvalError::MissingBinding(v))?;
                let coordinate = self
                    .law
                    .support
                    .get(i)
                    .ok_or(EvalError::MissingTableEntry)?
                    .iter()
                    .position(|&x| x == value);
                Ok((i, coordinate))
            })
            .collect::<Result<_, EvalError>>()?;
        self.tick(self.law.weights.len().saturating_mul(variables.len().max(1)))?;
        if constraints.iter().any(|(_, coordinate)| coordinate.is_none()) {
            return Ok(0.0);
        }
        Ok(self
            .law
            .weights
            .iter()
            .enumerate()
            .filter(|(cell, _)| {
                constraints.iter().all(|(i, coordinate)| {
                    (cell / self.strides[*i]) % self.law.support[*i].len() == coordinate.unwrap()
                })
            })
            .map(|(_, w)| w)
            .sum())
    }
}
impl DistributionProvider for Provider<'_> {
    fn probability(
        &self,
        factor: &FactorSpec<'_>,
        assignment: &Assignment,
        _: &EvalContext,
    ) -> Result<f64, EvalError> {
        if factor.domain != antecedent_expr::DomainRef::Observational
            || !factor.intervention.is_empty()
            || !factor.population.is_empty()
            || factor.regime.is_some()
        {
            return Err(EvalError::ProviderKind(
                "finite law only supplies single-study observational factors",
            ));
        }
        let denominator = self.mass(factor.conditioned_on, assignment)?;
        if denominator <= 0.0 {
            return Err(EvalError::DivisionByZero);
        }
        let mut variables = factor.variables.to_vec();
        variables.extend_from_slice(factor.conditioned_on);
        Ok(self.mass(&variables, assignment)? / denominator)
    }
    fn support(
        &self,
        variables: &[VariableId],
        _: &EvalContext,
    ) -> Result<Arc<[Arc<[Value]>]>, EvalError> {
        let mut rows = vec![vec![]];
        for v in variables {
            let values =
                self.law.support.get(v.raw() as usize).ok_or(EvalError::MissingTableEntry)?;
            self.tick(rows.len().saturating_mul(values.len()))?;
            rows = rows
                .iter()
                .flat_map(|row| {
                    values.iter().map(|&v| {
                        let mut r = row.clone();
                        r.push(Value::f64(v));
                        r
                    })
                })
                .collect();
        }
        Ok(rows.into_iter().map(Arc::from).collect())
    }
    fn outcome(
        &self,
        variable: VariableId,
        assignment: &Assignment,
        _: &EvalContext,
    ) -> Result<f64, EvalError> {
        assignment.get(variable).and_then(Value::as_f64).ok_or(EvalError::MissingBinding(variable))
    }
    fn n_draws(&self) -> Option<usize> {
        None
    }
}
/// Evaluate a retained program under a complete finite observational law.
///
/// Positivity failures remain located in their original assignment slots. This
/// numerical component does not authenticate the program's causal provenance.
/// # Errors
/// Malformed law/assignments, incompatible semantics, cancellation or admission refusal.
pub fn evaluate_finite_cyclic_functional(
    source: &CyclicFunctionalInputs<'_>,
    law: &FiniteCyclicLaw,
    assignments: &[Vec<(u32, f64)>],
    ctx: &ExecutionContext,
) -> Result<CyclicFunctionalReport, EstimationError> {
    let nodes = source.nodes;
    let continuous = source.continuous;
    let derivation = source.derivation.to_vec();
    let source_identity = source.identity;
    if continuous {
        return Err(invalid("finite-law consumption requires discrete identification semantics"));
    }
    if law.support.len() != nodes || assignments.is_empty() || assignments.len() > 1024 {
        return Err(invalid("finite law shape/assignment limit"));
    }
    let mut strides = Vec::new();
    let mut cells = 1usize;
    for values in &law.support {
        if values.is_empty()
            || values.len() > 64
            || values.iter().enumerate().any(|(i, v)| !v.is_finite() || values[..i].contains(v))
        {
            return Err(invalid("finite variable support"));
        }
        strides.push(cells);
        cells = cells.checked_mul(values.len()).ok_or_else(|| invalid("finite law size"))?;
        if cells > 65536 {
            return Err(invalid("finite law cell limit"));
        }
    }
    let total: f64 = law.weights.iter().sum();
    if law.weights.len() != cells
        || law.weights.iter().any(|v| !v.is_finite() || *v < 0.0)
        || !total.is_finite()
        || total <= 0.0
    {
        return Err(invalid("finite joint weights"));
    }
    let _lease = ctx.acquire_admission(1, (cells as u64).saturating_mul(256)).map_err(invalid)?;
    let checked = source.program.compile().map_err(invalid)?;
    let provider = Provider { law, strides, steps: Cell::new(0), ctx };
    let mut values = Vec::new();
    for pairs in assignments {
        let mut assignment = Assignment::new();
        for (i, &(v, x)) in pairs.iter().enumerate() {
            if v as usize >= nodes || !x.is_finite() || pairs[..i].iter().any(|(old, _)| *old == v)
            {
                return Err(invalid("functional assignment"));
            }
            assignment.set(VariableId::from_raw(v), Value::f64(x));
        }
        let (value, failure) =
            match checked.evaluate_with(&provider, &EvalContext::default(), &assignment) {
                Ok(v) if v.is_finite() => (Some(v), None),
                Ok(_) => (None, Some("nonfinite functional".into())),
                Err(error) => (None, Some(error.to_string())),
            };
        if ctx.cancellation.is_cancelled() || provider.steps.get() > 50_000_000 {
            return Err(invalid("functional cancellation/operation budget"));
        }
        values.push(CyclicFunctionalValue { assignment: pairs.clone(), value, failure });
    }
    Ok(CyclicFunctionalReport {
        source_identity: source_identity.into(),
        derivation,
        values,
        operations: provider.steps.get(),
    })
}
