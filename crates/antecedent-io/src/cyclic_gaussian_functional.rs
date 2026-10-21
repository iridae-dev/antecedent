// SPDX-License-Identifier: MIT OR Apache-2.0
//! Source authorization and Gaussian consumption of theorem-derived cyclic functionals.
use crate::FunctionalProgramWire;
use crate::cyclic_family_artifact::CyclicFamilyRequest;
use crate::cyclic_graph::CyclicGraphRequest;
use crate::cyclic_system_artifact::CyclicArtifactError;
use antecedent_core::ExecutionContext;
pub use antecedent_estimate::cyclic_gaussian_functional::{
    GaussianCyclicFunctionalReport, GaussianCyclicLaw,
};
fn invalid(s: impl std::fmt::Display) -> CyclicArtifactError {
    CyclicArtifactError::Invalid(s.to_string())
}
/// Recheck the theorem-derived IR, estimate an optional IID Gaussian density,
/// analytically marginalize its canonical factors, and evaluate original assignments.
/// Returns the common compiled family report, retaining its admitted output allocation.
/// # Errors
/// Source/IR mismatch, wrong semantics, nonintegrability, malformed law, cancellation or resources.
pub fn evaluate_gaussian_cyclic_functional(
    source: &CyclicGraphRequest,
    source_identity: &str,
    program: &FunctionalProgramWire,
    law: &GaussianCyclicLaw,
    assignments: &[Vec<(u32, f64)>],
    ctx: &ExecutionContext,
) -> Result<crate::cyclic_family_artifact::CyclicFamilyReport, CyclicArtifactError> {
    CyclicFamilyRequest::GaussianFunctional {
        source: source.clone(),
        source_identity: source_identity.into(),
        program: program.clone(),
        law: law.clone(),
        assignments: assignments.to_vec(),
    }
    .execute(ctx)
}

pub(crate) fn evaluate_gaussian_cyclic_functional_checked(
    source: &crate::cyclic_functional::CheckedCyclicFunctionalSource,
    law: &GaussianCyclicLaw,
    assignments: &[Vec<(u32, f64)>],
    frozen_density: Option<&crate::cyclic_family_artifact::FrozenGaussianDensity>,
    ctx: &ExecutionContext,
) -> Result<GaussianCyclicFunctionalReport, CyclicArtifactError> {
    antecedent_estimate::cyclic_gaussian_functional::evaluate_gaussian_cyclic_functional(
        &source.numerical_inputs(),
        law,
        assignments,
        frozen_density,
        ctx,
    )
    .map_err(invalid)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        cyclic_family_artifact::{CyclicFamilyAnswer, CyclicFamilyRequest, consume_cyclic_family},
        cyclic_graph::{CyclicGraphReport, CyclicGraphSpec},
    };
    #[test]
    fn continuous_frontdoor_matches_independent_latent_gaussian_truth_and_replays_fitting() {
        let ctx = ExecutionContext::for_tests(4);
        let source = CyclicGraphRequest::Identify {
            graph: CyclicGraphSpec {
                nodes: 3,
                directed: vec![(0, 1), (1, 2)],
                bidirected: vec![(0, 2)],
            },
            outcomes: vec![2],
            treatments: vec![0],
            continuous: true,
            max_steps: 1000,
            io_scm: true,
            scc_product_density: true,
        };
        let source_report =
            CyclicFamilyRequest::Graph { request: source.clone() }.execute(&ctx).unwrap();
        let CyclicFamilyAnswer::Graph(CyclicGraphReport::Identify {
            program: Some(program), ..
        }) = source_report.answer
        else {
            panic!("frontdoor identification")
        };
        let covariance = vec![2., 1., 1.5, 1., 1.5, 1.55, 1.5, 1.55, 3.01];
        let request = CyclicFamilyRequest::GaussianFunctional {
            source: source.clone(),
            source_identity: source_report.identity.clone(),
            program: program.clone(),
            law: GaussianCyclicLaw::Specified { mean: vec![0.; 3], covariance: covariance.clone() },
            assignments: vec![vec![(0, 1.), (2, 0.)], vec![(0, 1.)]],
        };
        let report = request.execute(&ctx).unwrap();
        let CyclicFamilyRequest::GaussianFunctional { law, assignments, .. } = &request else {
            unreachable!()
        };
        let public = evaluate_gaussian_cyclic_functional(
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
        let CyclicFamilyAnswer::GaussianFunctional(answer) = report.answer else { panic!() };
        let truth = (-0.5_f64 * 0.4 * 0.4 / 2.13).exp() / (2. * std::f64::consts::PI * 2.13).sqrt();
        assert!((answer.values[0].value.unwrap() - truth).abs() < 1e-12, "{answer:?} {truth}");
        assert!(answer.values[1].value.is_none());
        // Independent symmetric moment fixture: X=U+E, M=.5X+V,
        // Y=.8M+.7U+W. The observational MLE moments are exactly known.
        let mut rows = Vec::new();
        for mask in 0..16 {
            let e: Vec<_> = (0..4).map(|i| if mask & (1 << i) == 0 { -1. } else { 1. }).collect();
            let x = e[0] + e[1];
            let m = 0.5 * x + e[2];
            let y = 0.8 * m + 0.7 * e[0] + e[3];
            rows.push(vec![x, m, y]);
        }
        let fitted = CyclicFamilyRequest::GaussianFunctional {
            source,
            source_identity: source_report.identity,
            program,
            law: GaussianCyclicLaw::Fit { rows, independent_observational_replicates: true },
            assignments: vec![vec![(0, 1.), (2, 0.)]],
        };
        let observed =
            antecedent_core::execution_attempt::observe_with_context(&ctx, || fitted.export(&ctx));
        assert_eq!(observed.facts.as_ref().unwrap().observed_model_fits(), Some(1));
        assert_eq!(observed.facts.as_ref().unwrap().observed_scans(), Some(1));
        let (report, bytes) = observed.result.unwrap();
        let CyclicFamilyAnswer::GaussianFunctional(answer) = &report.answer else { panic!() };
        assert_eq!(answer.fitted_rows, Some(16));
        for (a, b) in answer.covariance.iter().zip(covariance) {
            assert!((a - b).abs() < 1e-12);
        }
        assert!((answer.values[0].value.unwrap() - truth).abs() < 1e-12);
        assert_eq!(consume_cyclic_family(&bytes, &report.identity, &ctx).unwrap(), report);
        let mut altered = fitted.clone();
        if let CyclicFamilyRequest::GaussianFunctional { source_identity, .. } = &mut altered {
            *source_identity = "0".repeat(64);
        }
        assert!(altered.execute(&ctx).is_err());
    }
}
