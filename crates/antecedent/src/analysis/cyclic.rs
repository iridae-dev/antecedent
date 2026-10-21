// SPDX-License-Identifier: MIT OR Apache-2.0
//! Nonlinear equilibrium specialization of the existing PreparedStudy handle.
//! Model/data/inference remain frozen; only intervention worlds can retarget
//! while retaining the fitted mechanisms. New data requires a new preparation.

use crate::analysis::{PreparedStudy, Study};
use antecedent_core::ExecutionContext;
use antecedent_io::cyclic_system_artifact::{
    CyclicArtifactError, CyclicExecutionReport, CyclicExecutionRequest, PreparedCyclicExecution,
};
use antecedent_model::nonlinear_equilibrium::Clamp;

/// Frozen native state of a cyclic specialization of `PreparedStudy`.
#[derive(Clone, Debug)]
pub struct CyclicPreparedState {
    product: PreparedCyclicExecution,
}

/// Inspection of a retained cyclic physical program.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CyclicPreparedInspection {
    /// Complete target/source/inference request identity.
    pub identity: String,
    /// Fits actually performed when constructing this handle (zero on retarget).
    pub fits_performed: usize,
    /// Per-world SCC solve schedules, sources before sinks.
    pub solve_blocks: Vec<Vec<Vec<usize>>>,
    /// Typed mechanism and simultaneous-equation execution plans.
    pub execution_plans: Vec<antecedent_model::compile::CompiledSystemPlan>,
}

impl Study {
    /// Prepare a specified or jointly fitted nonlinear cyclic model as a
    /// specialization of the common prepared handle. Existing sampled DAG routes
    /// and their graph invariants remain unchanged.
    ///
    /// # Errors
    /// Malformed programs, unsupported likelihoods, failed fits, cancellation or resource refusal.
    pub fn prepare_cyclic(
        request: CyclicExecutionRequest,
        ctx: &ExecutionContext,
    ) -> Result<PreparedStudy<CyclicPreparedState>, CyclicArtifactError> {
        let product = PreparedCyclicExecution::prepare(request, ctx)?;
        Ok(PreparedStudy { state: CyclicPreparedState { product } })
    }
}

impl PreparedStudy<CyclicPreparedState> {
    /// Borrow the complete frozen request, including aligned observations.
    #[must_use]
    pub fn request(&self) -> &CyclicExecutionRequest {
        self.state.product.request()
    }

    /// Inspect the retained target binding, actual solve blocks and fitting work.
    ///
    /// # Errors
    /// Invalid intervention bindings or an invalid retained execution schedule.
    pub fn inspect(&self) -> Result<CyclicPreparedInspection, CyclicArtifactError> {
        Ok(CyclicPreparedInspection {
            identity: self.state.product.identity()?,
            fits_performed: self.state.product.fits_performed(),
            solve_blocks: self.state.product.blocks()?,
            execution_plans: self.state.product.execution_plans()?,
        })
    }
    /// Execute from the retained physical product, performing zero additional fits.
    ///
    /// # Errors
    /// A refused numerical certificate, failed solve, cancellation or resource refusal.
    pub fn estimate(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<CyclicExecutionReport, CyclicArtifactError> {
        self.state.product.estimate(ctx)
    }
    /// Retarget prespecified worlds while preserving the trained mechanisms.
    ///
    /// # Errors
    /// Malformed target/noise/initial-state bindings or retained-memory refusal.
    pub fn retarget_worlds(
        &self,
        worlds: Vec<Vec<Clamp>>,
        noise: Vec<Vec<f64>>,
        initial: Vec<f64>,
    ) -> Result<Self, CyclicArtifactError> {
        Ok(Self {
            state: CyclicPreparedState {
                product: self.state.product.retarget(worlds, noise, initial)?,
            },
        })
    }
    /// Export a result without refitting; the consumer independently refits/replays.
    ///
    /// # Errors
    /// Invalid artifact requests, failed numerical execution, cancellation or resource refusal.
    pub fn export(
        &self,
        artifact_id: &str,
        ctx: &ExecutionContext,
    ) -> Result<(CyclicExecutionReport, Vec<u8>), CyclicArtifactError> {
        self.state.product.export(artifact_id, ctx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_estimate::cyclic_likelihood::{CyclicFitOptions, CyclicParameter};
    use antecedent_io::cyclic_system_artifact::{CyclicFitRequest, consume_cyclic_artifact};
    use antecedent_model::nonlinear_equilibrium::{
        Activation, Equation, EquilibriumModel, RegimeRow, SolvePolicy,
    };
    #[test]
    fn common_prepared_handle_retains_fit_retargets_and_independently_replays() {
        let request = CyclicExecutionRequest {
            model: EquilibriumModel {
                equations: vec![
                    Equation {
                        id: "a".into(),
                        intercept: 0.0,
                        weights: vec![0.0, 0.4],
                        activation: Activation::Identity,
                        expression: None,
                        parameters: vec![],
                    },
                    Equation {
                        id: "b".into(),
                        intercept: 0.0,
                        weights: vec![0.5, 0.0],
                        activation: Activation::Identity,
                        expression: None,
                        parameters: vec![],
                    },
                ],
                noise_covariance: vec![1.0, 0.0, 0.0, 1.0],
            },
            fit: Some(CyclicFitRequest {
                parameters: vec![CyclicParameter::Weight { equation: 1, parent: 0 }],
                rows: vec![
                    RegimeRow {
                        values: vec![1.0, 0.7],
                        clamps: vec![Clamp { variable: 0, value: 1.0 }],
                    },
                    RegimeRow {
                        values: vec![-1.0, -0.7],
                        clamps: vec![Clamp { variable: 0, value: -1.0 }],
                    },
                ],
                options: CyclicFitOptions::default(),
            }),
            worlds: vec![vec![]],
            noise: vec![vec![0.0, 0.0]],
            initial: vec![0.0; 2],
            policy: SolvePolicy::default(),
            factual: None,
        };
        let ctx = ExecutionContext::for_tests(1);
        let prepared = Study::prepare_cyclic(request, &ctx).unwrap();
        assert_eq!(prepared.inspect().unwrap().fits_performed, 1);
        let original = prepared.estimate(&ctx).unwrap();
        let retargeted = prepared
            .retarget_worlds(
                vec![vec![Clamp { variable: 0, value: 2.0 }]],
                vec![vec![0.0, 0.0]],
                vec![0.0; 2],
            )
            .unwrap();
        assert_eq!(retargeted.inspect().unwrap().fits_performed, 0);
        let (answer, bytes) = retargeted.export("retargeted", &ctx).unwrap();
        assert_eq!(answer.fit, original.fit);
        assert_ne!(answer.identity, original.identity);
        assert!((answer.worlds[0][0].values[1] - 1.4).abs() < 1e-4);
        assert_eq!(consume_cyclic_artifact(&bytes, &answer.identity, &ctx).unwrap(), answer);
    }
}

/// Immutable retained answer/artifact of a typed cyclic family preparation.
#[derive(Clone, Debug)]
pub struct CyclicFamilyPreparedState {
    pub(super) report: std::sync::Arc<antecedent_io::cyclic_family_artifact::CyclicFamilyReport>,
    pub(super) request: std::sync::Arc<antecedent_io::cyclic_family_artifact::CyclicFamilyRequest>,
    pub(super) artifact: std::sync::Arc<[u8]>,
    family: String,
    retained_bytes: u64,
    _retention: std::sync::Arc<antecedent_core::execution::AdmissionLease>,
}
/// Inspection of a typed family preparation, including retained work.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CyclicFamilyPreparedInspection {
    /// Complete source/target/semantics/numerical-policy identity.
    pub identity: String,
    /// Explicit equilibrium/dynamics/equation/graph operator family.
    pub family: String,
    /// One full execution constructs the immutable product; reuse performs zero.
    pub preparation_evaluations: usize,
    /// Reserved retained payload/answer bytes, released with the last handle.
    pub retained_bytes: u64,
}
impl Study {
    /// Prepare any typed cyclic family on the common retained handle. Statistical
    /// fitting, abduction or numerical execution occurs once. New source/data,
    /// target, initial law or policy requires a new preparation and identity.
    ///
    /// # Errors
    /// Malformed family programs, unsupported mathematical conditions, failed execution, cancellation or resource refusal.
    pub fn prepare_cyclic_family(
        request: &antecedent_io::cyclic_family_artifact::CyclicFamilyRequest,
        ctx: &ExecutionContext,
    ) -> Result<PreparedStudy<CyclicFamilyPreparedState>, CyclicArtifactError> {
        let family = request.family_name().to_owned();
        let (report, artifact) = request.export(ctx)?;
        let retained_bytes = (antecedent_io::cyclic_system_artifact::cyclic_serialized_size(
            &(request, &report),
            ctx,
        )? as u64)
            .saturating_mul(8)
            .saturating_add((artifact.len() as u64) * 4 + 4096);
        let retention = ctx
            .acquire_admission(0, retained_bytes)
            .map_err(|e| CyclicArtifactError::Invalid(e.to_string()))?;
        Ok(PreparedStudy {
            state: CyclicFamilyPreparedState {
                request: std::sync::Arc::new(request.clone()),
                report: std::sync::Arc::new(report),
                artifact: artifact.into(),
                family,
                retained_bytes,
                _retention: std::sync::Arc::new(retention),
            },
        })
    }
}
impl PreparedStudy<CyclicFamilyPreparedState> {
    /// Inspect the actual immutable family product without another fit/solve.
    #[must_use]
    ///
    /// # Errors
    /// Invalid intervention bindings or an invalid retained execution schedule.
    pub fn inspect(&self) -> CyclicFamilyPreparedInspection {
        CyclicFamilyPreparedInspection {
            identity: self.state.report.identity.clone(),
            family: self.state.family.clone(),
            preparation_evaluations: 1,
            retained_bytes: self.state.retained_bytes,
        }
    }
    /// Reuse the retained answer and independent artifact; enforce cancellation
    /// and output-copy admission even when no causal fitting is repeated.
    ///
    /// # Errors
    /// Invalid artifact requests, failed numerical execution, cancellation or resource refusal.
    pub fn export(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<
        (antecedent_io::cyclic_family_artifact::CyclicFamilyReport, Vec<u8>),
        CyclicArtifactError,
    > {
        let _output = ctx
            .acquire_admission(0, self.state.retained_bytes)
            .map_err(|e| CyclicArtifactError::Invalid(e.to_string()))?;
        Ok((self.state.report.as_ref().clone(), self.state.artifact.to_vec()))
    }
}
