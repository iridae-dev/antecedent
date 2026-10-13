//! Assembly of the joint parameter covariance of a composition.
//!
//! Each mechanism's own fit reports only its marginal covariance. Whether the
//! mechanisms' estimates are independent, correlated by a declared amount, or
//! unknown is a declaration of the composition, never a default. This module
//! turns the declarations into one parameter covariance, or refuses.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;

use super::coherence::{
    CoherenceFinding, CoherenceKind, is_positive_semidefinite, parameter_layout,
    uncertainty_findings,
};
use super::descriptor::{ComposedModelSpec, ParameterId, PosteriorDependence};

/// The assembled parameter law of a composition.
#[derive(Clone, Debug, PartialEq)]
pub struct ParameterCovariance {
    /// Parameters in canonical order: mechanisms by variable, then intercept,
    /// then coefficients by parent.
    pub parameters: Vec<ParameterId>,
    /// Row-major covariance over `parameters`.
    pub covariance: Vec<f64>,
    /// How the cross-mechanism dependence was licensed.
    pub dependence_label: &'static str,
}

/// Assemble the joint parameter covariance of a canonical specification.
///
/// # Errors
/// Every finding that blocks an uncertainty claim: a mechanism without a
/// reported marginal, unknown dependence over more than one mechanism, an
/// unpaired block, or an assembled matrix that is not positive semidefinite.
pub fn assemble_parameter_covariance(
    spec: &ComposedModelSpec,
) -> Result<ParameterCovariance, Vec<CoherenceFinding>> {
    let spec = spec.canonical();
    let blocking = uncertainty_findings(&spec);
    if !blocking.is_empty() {
        return Err(blocking);
    }
    let layout = parameter_layout(&spec);
    let p = layout.len();
    let index: BTreeMap<&ParameterId, usize> =
        layout.iter().enumerate().map(|(i, id)| (id, i)).collect();
    let mut cov = vec![0.0_f64; p * p];
    let label = match &spec.dependence {
        PosteriorDependence::Joint { parameters, covariance, .. } => {
            let k = parameters.len();
            for (i, a) in parameters.iter().enumerate() {
                for (j, b) in parameters.iter().enumerate() {
                    if let (Some(&ia), Some(&ib)) = (index.get(a), index.get(b)) {
                        cov[ia * p + ib] = covariance[i * k + j];
                    }
                }
            }
            "aligned_joint_law"
        }
        other => {
            for d in &spec.mechanisms {
                let Some(u) = &d.uncertainty else { continue };
                let k = u.parameters.len();
                for (i, a) in u.parameters.iter().enumerate() {
                    for (j, b) in u.parameters.iter().enumerate() {
                        let ida = ParameterId { mechanism: d.variable.clone(), term: a.clone() };
                        let idb = ParameterId { mechanism: d.variable.clone(), term: b.clone() };
                        if let (Some(&ia), Some(&ib)) = (index.get(&ida), index.get(&idb)) {
                            cov[ia * p + ib] = u.covariance[i * k + j];
                        }
                    }
                }
            }
            match other {
                PosteriorDependence::CrossCovariance { blocks } => {
                    for block in blocks {
                        let cols = block.second_parameters.len();
                        for (i, a) in block.first_parameters.iter().enumerate() {
                            for (j, b) in block.second_parameters.iter().enumerate() {
                                let ida =
                                    ParameterId { mechanism: block.first.clone(), term: a.clone() };
                                let idb = ParameterId {
                                    mechanism: block.second.clone(),
                                    term: b.clone(),
                                };
                                if let (Some(&ia), Some(&ib)) = (index.get(&ida), index.get(&idb)) {
                                    let value = block.values[i * cols + j];
                                    cov[ia * p + ib] = value;
                                    cov[ib * p + ia] = value;
                                }
                            }
                        }
                    }
                    "declared_cross_covariance"
                }
                PosteriorDependence::IndependentByDesign { .. } => "declared_independent_by_design",
                _ => "single_mechanism",
            }
        }
    };
    if !is_positive_semidefinite(&cov, p) {
        let finding = CoherenceFinding::new(
            CoherenceKind::UncertaintyAlignment,
            "mechanism_composition.uncertainty_assembled_not_psd",
            "composition",
            "the declared marginals and cross-covariances do not form one valid joint law; supply a positive semidefinite joint covariance",
        );
        debug_assert_eq!(
            finding.kind.reason_code(),
            antecedent_core::reason_code!("mechanism_composition_incoherent")
        );
        return Err(vec![finding]);
    }
    Ok(ParameterCovariance { parameters: layout, covariance: cov, dependence_label: label })
}
