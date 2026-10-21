// SPDX-License-Identifier: MIT OR Apache-2.0
//! Source authorization and finite-law consumption of theorem-derived cyclic functionals.
//! Re-identification binds the serialized program to its graph and assumptions.
use crate::cyclic_family_artifact::CyclicFamilyRequest;
use crate::cyclic_graph::{CyclicGraphReport, CyclicGraphRequest};
use crate::cyclic_system_artifact::CyclicArtifactError;
use crate::{FunctionalProgramWire, functional_program_from_wire};
use antecedent_core::ExecutionContext;
pub use antecedent_estimate::cyclic_functional::{
    CyclicFunctionalReport, CyclicFunctionalValue, FiniteCyclicLaw,
};
use antecedent_expr::ProgramLimits;

fn invalid(s: impl std::fmt::Display) -> CyclicArtifactError {
    CyclicArtifactError::Invalid(s.to_string())
}
#[derive(Clone, Debug)]
pub(crate) struct CheckedCyclicFunctionalSource {
    pub(crate) nodes: usize,
    pub(crate) continuous: bool,
    pub(crate) derivation: Vec<String>,
    pub(crate) identity: String,
    pub(crate) calculus: bool,
    pub(crate) program: antecedent_expr::FunctionalProgram,
}
impl CheckedCyclicFunctionalSource {
    pub(crate) fn numerical_inputs(
        &self,
    ) -> antecedent_estimate::cyclic_functional::CyclicFunctionalInputs<'_> {
        antecedent_estimate::cyclic_functional::CyclicFunctionalInputs {
            nodes: self.nodes,
            continuous: self.continuous,
            derivation: &self.derivation,
            identity: &self.identity,
            calculus: self.calculus,
            program: &self.program,
        }
    }
}
pub(crate) fn checked_functional_source(
    source: &CyclicGraphRequest,
    source_identity: &str,
    program: &FunctionalProgramWire,
    ctx: &ExecutionContext,
) -> Result<CheckedCyclicFunctionalSource, CyclicArtifactError> {
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
    let checked =
        functional_program_from_wire(program, ProgramLimits::default()).map_err(invalid)?;
    Ok(CheckedCyclicFunctionalSource {
        nodes: graph.nodes as usize,
        continuous,
        derivation,
        identity: source_identity.into(),
        calculus: matches!(source, CyclicGraphRequest::Calculus { .. }),
        program: checked,
    })
}
/// Recheck identification and consume its functional using a bounded finite law.
/// Positive denominators are checked at evaluation, separately from graphical identification.
/// Returns the common compiled family report, retaining its admitted output allocation.
pub fn evaluate_cyclic_functional(
    source: &CyclicGraphRequest,
    source_identity: &str,
    program: &FunctionalProgramWire,
    law: &FiniteCyclicLaw,
    assignments: &[Vec<(u32, f64)>],
    ctx: &ExecutionContext,
) -> Result<crate::cyclic_family_artifact::CyclicFamilyReport, CyclicArtifactError> {
    CyclicFamilyRequest::Functional {
        source: source.clone(),
        source_identity: source_identity.into(),
        program: program.clone(),
        law: law.clone(),
        assignments: assignments.to_vec(),
    }
    .execute(ctx)
}

pub(crate) fn evaluate_cyclic_functional_checked(
    source: &CheckedCyclicFunctionalSource,
    law: &FiniteCyclicLaw,
    assignments: &[Vec<(u32, f64)>],
    ctx: &ExecutionContext,
) -> Result<CyclicFunctionalReport, CyclicArtifactError> {
    antecedent_estimate::cyclic_functional::evaluate_finite_cyclic_functional(
        &source.numerical_inputs(),
        law,
        assignments,
        ctx,
    )
    .map_err(invalid)
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
        let CyclicFamilyRequest::Functional { law, assignments, .. } = &request else {
            unreachable!()
        };
        let public = evaluate_cyclic_functional(
            &source,
            &source_report.identity,
            &program,
            law,
            assignments,
            &ctx,
        )
        .unwrap();
        assert_eq!(public, report);
        assert!(public.retained_admitted_bytes() > 0);
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
