//! Sensitivity ranges of a vector of compiled results, one coordinate per result.
//!
//! [`sensitivity_vector_range`] reads several compiled results, each checked against the
//! prepared handle that produced it exactly as [`PreparedStudy::sensitivity_range`] checks one,
//! as the coordinates of one vector target: the outcomes of a treatment (several average-effect
//! studies) or of one mediator (several mediation studies). Each coordinate's summary is the one
//! the scalar analysis computes on that study's own columns, and the vector range reads them at
//! the weakest coordinate:
//!
//! * [`SensitivityVectorRequest::EValue`]: [`evalue_vector_range`] on each result's standardized
//!   effect and standard error;
//! * [`SensitivityVectorRequest::OmittedVariableBias`]: [`ovb_vector_range`] on each study's
//!   linear-adjustment summary;
//! * [`SensitivityVectorRequest::MediationComponent`] and
//!   [`SensitivityVectorRequest::MediationPath`]: [`mediation_vector_range`] and
//!   [`mediation_path_ovb_range`] on each study's working-model summaries;
//! * [`SensitivityVectorRequest::MediationComponentEValue`] and
//!   [`SensitivityVectorRequest::MediationComponentOvb`]: [`component_evalue_range`] on each
//!   mediation result's standardized requested contrast and [`component_ovb_range`] on each
//!   study's mediator-outcome coefficient, bound to the mediation assumption.
//!
//! The vector target is labelled by the coordinates' outcome names and defined by the
//! identities of their lowered programs, in order. Its family is the mediation family for the
//! mediation analyses and the joint-contrast family otherwise.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{CanonicalWriter, ExecutionContext};
use antecedent_validate::target_native::{RefutationTarget, TargetFamily};
use antecedent_validate::{
    AssumptionRange, ComponentModel, EValueBounds, EffectInterval, EffectScale,
    LinearAdjustmentSummary, MediationComponent, MediationComponents, MediationPath,
    MediationPaths, component_evalue_range, component_ovb_range, evalue_vector_range,
    mediation_components_fit, mediation_path_ovb_range, mediation_paths_fit,
    mediation_vector_range, ovb_vector_range,
};

use super::super::prepared::PreparedStudy;
use super::super::target_validation::result_quadrant;
use super::{Bound, Standardized};
use crate::error::CausalError;
use crate::result::StudyResult;

/// Which sensitivity analysis to run on a vector of compiled results.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum SensitivityVectorRequest {
    /// E-values of every coordinate's standardized effect, read at the weakest.
    EValue,
    /// Partial-R² sensitivity of every coordinate's linear-adjustment coefficient.
    OmittedVariableBias {
        /// Share of the estimate a confounder must remove to explain it away.
        q: f64,
        /// Level of the significance robustness values and of the adjusted limits.
        alpha: f64,
        /// Outcome partial R² of the extreme-scenario curve, in `(0, 1]`.
        r2yz_extreme: f64,
    },
    /// The indirect or direct effect of every coordinate under the mediator-outcome error
    /// correlation.
    MediationComponent(MediationComponent),
    /// Partial-R² sensitivity of one path of every coordinate's mediation.
    MediationPath {
        /// The path.
        path: MediationPath,
        /// Share of the estimate a confounder must remove to explain it away.
        q: f64,
        /// Level of the significance robustness values and of the adjusted limits.
        alpha: f64,
        /// Outcome partial R² of the extreme-scenario curve, in `(0, 1]`.
        r2yz_extreme: f64,
    },
    /// E-values of every mediation result's requested contrast, bound to the mediation
    /// assumption.
    MediationComponentEValue,
    /// Partial-R² sensitivity of every coordinate's mediator-outcome coefficient, bound to the
    /// mediation assumption.
    MediationComponentOvb {
        /// Share of the estimate a confounder must remove to explain it away.
        q: f64,
        /// Level of the significance robustness values and of the adjusted limits.
        alpha: f64,
        /// Outcome partial R² of the extreme-scenario curve, in `(0, 1]`.
        r2yz_extreme: f64,
    },
}

/// One sensitivity analysis of a vector of compiled results.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum SensitivityVectorOutcome {
    /// [`SensitivityVectorRequest::EValue`] and
    /// [`SensitivityVectorRequest::MediationComponentEValue`].
    EValue {
        /// Per coordinate: the standardized effect, its standard error and the residual
        /// standard deviation it is standardized by.
        standardized: Vec<(f64, Option<f64>, f64)>,
        /// Per coordinate E-values.
        bounds: Vec<EValueBounds>,
        /// The range.
        range: AssumptionRange,
    },
    /// [`SensitivityVectorRequest::OmittedVariableBias`] and
    /// [`SensitivityVectorRequest::MediationComponentOvb`].
    OmittedVariableBias {
        /// Per coordinate summaries.
        summaries: Vec<LinearAdjustmentSummary>,
        /// The range.
        range: AssumptionRange,
    },
    /// [`SensitivityVectorRequest::MediationComponent`].
    MediationComponent {
        /// Per coordinate regression summaries.
        components: Vec<MediationComponents>,
        /// The range.
        range: AssumptionRange,
    },
    /// [`SensitivityVectorRequest::MediationPath`].
    MediationPath {
        /// Per coordinate path summaries.
        paths: Vec<MediationPaths>,
        /// The range.
        range: AssumptionRange,
    },
}

impl SensitivityVectorOutcome {
    /// The range.
    #[must_use]
    pub const fn range(&self) -> &AssumptionRange {
        match self {
            Self::EValue { range, .. }
            | Self::OmittedVariableBias { range, .. }
            | Self::MediationComponent { range, .. }
            | Self::MediationPath { range, .. } => range,
        }
    }
}

/// Run `request` on `coordinates`, each a result and the prepared handle that produced it from
/// its retained table.
///
/// # Errors
///
/// No coordinate; any refusal of [`PreparedStudy::sensitivity_range`] for one coordinate (a
/// result another program produced or that was modified, a non-mediation study for the mediation
/// analyses, a 0/1 outcome for the E-values); and the refusals of the kernel function.
pub fn sensitivity_vector_range(
    coordinates: &[(&PreparedStudy, &StudyResult)],
    request: &SensitivityVectorRequest,
    ctx: &ExecutionContext,
) -> Result<SensitivityVectorOutcome, CausalError> {
    let Some(&(_, first)) = coordinates.first() else {
        return Err(CausalError::Unsupported {
            message: "a vector sensitivity range needs at least one coordinate",
        });
    };
    let mut bound: Vec<(&PreparedStudy, Bound<'_>)> = Vec::with_capacity(coordinates.len());
    let mut labels = Vec::with_capacity(coordinates.len());
    let mut writer = CanonicalWriter::new("antecedent.sensitivity_vector_target");
    for &(study, result) in coordinates {
        let coordinate = study.bind(result, ctx)?;
        writer.bytes(&coordinate.program);
        labels.push(
            study
                .schema()
                .variables()
                .iter()
                .find(|variable| variable.id == result.outcome)
                .map_or_else(
                    || format!("v{}", result.outcome.raw()),
                    |variable| variable.name.to_string(),
                ),
        );
        bound.push((study, coordinate));
    }
    let source = writer.finish();
    let mediation = matches!(
        request,
        SensitivityVectorRequest::MediationComponent(_)
            | SensitivityVectorRequest::MediationPath { .. }
            | SensitivityVectorRequest::MediationComponentEValue
            | SensitivityVectorRequest::MediationComponentOvb { .. }
    );
    let family = if mediation { TargetFamily::Mediation } else { TargetFamily::JointContrast };
    let target = RefutationTarget::new(family, result_quadrant(first), labels, source)
        .map_err(CausalError::from)?;
    match request {
        SensitivityVectorRequest::EValue | SensitivityVectorRequest::MediationComponentEValue => {
            let mut standardized = Vec::with_capacity(bound.len());
            for (_, coordinate) in &bound {
                let effect = if mediation {
                    coordinate.result.mediation().map(|m| &m.effect)
                } else {
                    coordinate.result.estimate.as_effect()
                }
                .ok_or(CausalError::Unsupported {
                    message: "the E-values read every coordinate's effect",
                })?;
                standardized.push(coordinate.standardized(effect)?);
            }
            let intervals: Vec<EffectInterval> = standardized
                .iter()
                .map(|s| EffectInterval::Standardized {
                    estimate: s.effect,
                    standard_error: s.standard_error,
                })
                .collect();
            let scale = EffectScale::StandardizedMeanDifference;
            let (bounds, range) = if mediation {
                component_evalue_range(target, &intervals, scale, ComponentModel::Mediation)?
            } else {
                evalue_vector_range(target, &intervals, scale)?
            };
            let standardized = standardized
                .iter()
                .map(|&Standardized { effect, standard_error, residual_sd }| {
                    (effect, standard_error, residual_sd)
                })
                .collect();
            Ok(SensitivityVectorOutcome::EValue { standardized, bounds, range })
        }
        SensitivityVectorRequest::OmittedVariableBias { q, alpha, r2yz_extreme } => {
            let summaries = bound
                .iter()
                .map(|(_, coordinate)| coordinate.linear_fit().map(|(fit, _)| fit.summary))
                .collect::<Result<Vec<_>, _>>()?;
            let range = ovb_vector_range(target, &summaries, *q, *alpha, *r2yz_extreme)?;
            Ok(SensitivityVectorOutcome::OmittedVariableBias { summaries, range })
        }
        SensitivityVectorRequest::MediationComponent(component) => {
            let mut components = Vec::with_capacity(bound.len());
            for (study, coordinate) in &bound {
                let (t, m, y, covariates) = study.mediation_columns(coordinate)?;
                components.push(mediation_components_fit(&t, &m, &y, &covariates)?);
            }
            let range = mediation_vector_range(target, &components, *component)?;
            Ok(SensitivityVectorOutcome::MediationComponent { components, range })
        }
        SensitivityVectorRequest::MediationPath { path, q, alpha, r2yz_extreme } => {
            let paths = mediation_paths(&bound)?;
            let range = mediation_path_ovb_range(target, &paths, *path, *q, *alpha, *r2yz_extreme)?;
            Ok(SensitivityVectorOutcome::MediationPath { paths, range })
        }
        SensitivityVectorRequest::MediationComponentOvb { q, alpha, r2yz_extreme } => {
            let summaries: Vec<LinearAdjustmentSummary> =
                mediation_paths(&bound)?.iter().map(|paths| paths.mediator_outcome).collect();
            let range = component_ovb_range(
                target,
                &summaries,
                *q,
                *alpha,
                *r2yz_extreme,
                ComponentModel::Mediation,
            )?;
            Ok(SensitivityVectorOutcome::OmittedVariableBias { summaries, range })
        }
    }
}

fn mediation_paths(
    bound: &[(&PreparedStudy, Bound<'_>)],
) -> Result<Vec<MediationPaths>, CausalError> {
    bound
        .iter()
        .map(|(study, coordinate)| {
            let (t, m, y, covariates) = study.mediation_columns(coordinate)?;
            mediation_paths_fit(&t, &m, &y, &covariates).map_err(CausalError::from)
        })
        .collect()
}
