// SPDX-License-Identifier: MIT OR Apache-2.0
//! Checked finite-law consumption of a theorem-derived cyclic functional.
//! Re-identification binds the serialized program to its graph and assumptions;
//! a user-supplied expression never acquires an identification claim by loading.
use crate::cyclic_family_artifact::CyclicFamilyRequest;
use crate::cyclic_graph::{CyclicGraphReport, CyclicGraphRequest};
use crate::cyclic_system_artifact::CyclicArtifactError;
use crate::{FunctionalProgramWire, functional_program_from_wire};
use antecedent_core::{ExecutionContext, Value, VariableId};
use antecedent_expr::{
    Assignment, DistributionProvider, EvalContext, EvalError, FactorSpec, ProgramLimits,
};
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
fn invalid(s: impl std::fmt::Display) -> CyclicArtifactError {
    CyclicArtifactError::Invalid(s.to_string())
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
pub(crate) fn checked_functional_source(
    source: &CyclicGraphRequest,
    source_identity: &str,
    program: &FunctionalProgramWire,
    ctx: &ExecutionContext,
) -> Result<(usize, bool, Vec<String>), CyclicArtifactError> {
    let (graph, continuous) = match source {
        CyclicGraphRequest::Identify { graph, continuous, .. }
        | CyclicGraphRequest::Adjust { graph, continuous, .. } => (graph, *continuous),
        CyclicGraphRequest::Calculus { graph, w, .. } if w.is_empty() => (graph, false),
        _ => {
            return Err(invalid(
                "finite functional source must produce an observational executable",
            ));
        }
    };
    let source_request = CyclicFamilyRequest::Graph { request: source.clone() };
    let source_report = source_request.execute(ctx)?;
    if source_report.identity != source_identity {
        return Err(invalid("identification source identity differs"));
    }
    let (expected, derivation) = match source_report.answer {
        crate::cyclic_family_artifact::CyclicFamilyAnswer::Graph(CyclicGraphReport::Identify {
            status,
            program: Some(program),
            derivation,
            ..
        }) if status == "identified" => (program, derivation),
        crate::cyclic_family_artifact::CyclicFamilyAnswer::Graph(CyclicGraphReport::Calculus {
            status,
            fully_observational: true,
            program: Some(program),
            derivation,
            ..
        }) if status == "verified_rule" => (program, derivation),
        _ => return Err(invalid("source did not produce a verified observational functional")),
    };
    if expected != *program {
        return Err(invalid("functional differs from theorem-derived source"));
    }
    Ok((graph.nodes as usize, continuous, derivation))
}
/// Recheck identification and consume its functional using a bounded finite law.
/// Positive denominators are checked at evaluation, separately from graphical identification.
pub fn evaluate_cyclic_functional(
    source: &CyclicGraphRequest,
    source_identity: &str,
    program: &FunctionalProgramWire,
    law: &FiniteCyclicLaw,
    assignments: &[Vec<(u32, f64)>],
    ctx: &ExecutionContext,
) -> Result<CyclicFunctionalReport, CyclicArtifactError> {
    let (nodes, continuous, derivation) =
        checked_functional_source(source, source_identity, program, ctx)?;
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
    let checked = functional_program_from_wire(program, ProgramLimits::default())
        .map_err(invalid)?
        .compile()
        .map_err(invalid)?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        cyclic_family_artifact::{CyclicFamilyAnswer, consume_cyclic_family},
        cyclic_graph::CyclicGraphSpec,
    };
    #[test]
    fn identified_functional_is_bound_evaluated_and_replayed_with_positivity_failures() {
        let ctx = ExecutionContext::for_tests(2);
        let source = CyclicGraphRequest::Identify {
            graph: CyclicGraphSpec {
                nodes: 3,
                directed: vec![(0, 1), (1, 2), (2, 1)],
                bidirected: vec![],
            },
            outcomes: vec![2],
            treatments: vec![0],
            continuous: false,
            max_steps: 100,
            io_scm: true,
            scc_product_density: true,
        };
        let source_report =
            CyclicFamilyRequest::Graph { request: source.clone() }.execute(&ctx).unwrap();
        let CyclicFamilyAnswer::Graph(CyclicGraphReport::Identify {
            program: Some(program), ..
        }) = source_report.answer
        else {
            panic!("no identified program")
        };
        let mut weights = vec![0.0; 8];
        for (cell, weight) in weights.iter_mut().enumerate() {
            let t = cell & 1 != 0;
            let a = cell & 2 != 0;
            let b = cell & 4 != 0;
            let pt = if t { 0.6 } else { 0.4 };
            let pa = if t { 0.8 } else { 0.2 };
            let pb = 0.1 + if a { 0.5 } else { 0.0 };
            *weight = pt * (if a { pa } else { 1.0 - pa }) * (if b { pb } else { 1.0 - pb });
        }
        let request = CyclicFamilyRequest::Functional {
            source: source.clone(),
            source_identity: source_report.identity.clone(),
            program: program.clone(),
            law: FiniteCyclicLaw { support: vec![vec![0.0, 1.0]; 3], weights },
            assignments: vec![
                vec![(0, 0.0), (2, 1.0)],
                vec![(0, 1.0), (2, 1.0)],
                vec![(0, 2.0), (2, 1.0)],
            ],
        };
        let (report, artifact) = request.export(&ctx).unwrap();
        let CyclicFamilyAnswer::Functional(answer) = &report.answer else { panic!("wrong answer") };
        assert!((answer.values[0].value.unwrap() - 0.2).abs() < 1e-12);
        assert!((answer.values[1].value.unwrap() - 0.5).abs() < 1e-12);
        assert!(answer.values[2].value.is_none());
        assert!(answer.values[2].failure.is_some());
        assert_eq!(consume_cyclic_family(&artifact, &report.identity, &ctx).unwrap(), report);
        let CyclicFamilyRequest::Functional { law, assignments, .. } = request else {
            unreachable!()
        };
        assert!(
            evaluate_cyclic_functional(
                &source,
                &"0".repeat(64),
                &program,
                &law,
                &assignments,
                &ctx
            )
            .is_err()
        );
    }
}
