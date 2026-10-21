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
    /// Fits actually performed by execution on this handle (zero before execution).
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
    /// Execute from the retained physical product, reusing compatible fitted mechanisms.
    ///
    /// # Errors
    /// A refused numerical certificate, failed solve, cancellation or resource refusal.
    pub fn estimate(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<CyclicExecutionReport, CyclicArtifactError> {
        self.state.product.estimate(ctx)
    }
    /// Execute with native facts on success, refusal or cancellation.
    #[must_use]
    pub fn estimate_observed(
        &self,
        ctx: &ExecutionContext,
    ) -> antecedent_core::execution_attempt::ExecutionAttempt<
        CyclicExecutionReport,
        CyclicArtifactError,
    > {
        antecedent_core::execution_attempt::observe_with_context(ctx, || {
            self.state.product.estimate(ctx)
        })
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
        let prepared = Study::prepare_cyclic(request.clone(), &ctx).unwrap();
        assert_eq!(prepared.inspect().unwrap().fits_performed, 0);
        let original = prepared.estimate(&ctx).unwrap();
        assert_eq!(prepared.inspect().unwrap().fits_performed, 1);
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

        use antecedent_core::execution_attempt::Operation;
        use antecedent_io::cyclic_family_artifact::{
            CyclicFamilyAnswer, CyclicFamilyRequest, consume_cyclic_family,
        };
        let fit = request.fit.unwrap();
        let mut family = CyclicFamilyRequest::FrequentistBootstrap {
            design: Default::default(),
            model: request.model,
            parameters: fit.parameters,
            rows: fit.rows,
            policy: fit.options,
            replicates: 2,
            seed: 23,
            worlds: request.worlds,
            noise: request.noise,
            initial: request.initial,
            solve_policy: request.policy,
        };
        let preparation = antecedent_core::execution_attempt::observe_with_context(&ctx, || {
            Study::prepare_cyclic_family(&family, &ctx)
        });
        assert_eq!(preparation.report.counts(Operation::AdjustedFit).attempted, 0);
        let compiled = preparation.result.unwrap();
        assert_eq!(compiled.inspect().preparation_evaluations, 0);
        let first = compiled.estimate_observed(&ctx);
        assert_eq!(first.report.counts(Operation::AdjustedFit).attempted, 2);
        let first = first.result.unwrap();
        if let CyclicFamilyRequest::FrequentistBootstrap { worlds, .. } = &mut family {
            *worlds = vec![vec![Clamp { variable: 0, value: 2.0 }]];
        }
        let rebound = compiled.rebind(&family, &ctx).unwrap();
        drop(compiled);
        let reused = rebound.estimate_observed(&ctx);
        assert_eq!(reused.report.counts(Operation::AdjustedFit).attempted, 0);
        let reused = reused.result.unwrap();
        if let (
            CyclicFamilyAnswer::FrequentistBootstrap { parameters: a, .. },
            CyclicFamilyAnswer::FrequentistBootstrap { parameters: b, .. },
        ) = (&first.answer, &reused.answer)
        {
            assert_eq!(a, b);
        } else {
            panic!("expected retained bootstrap draws");
        }
        let (exported, bytes) = rebound.export(&ctx).unwrap();
        assert_eq!(consume_cyclic_family(&bytes, &exported.identity, &ctx).unwrap(), exported);
        if let CyclicFamilyRequest::FrequentistBootstrap { rows, .. } = &mut family {
            for row in rows {
                row.values[1] *= 0.5;
            }
        }
        let refreshed = rebound.rebind(&family, &ctx).unwrap();
        let changed = refreshed.estimate_observed(&ctx);
        assert_eq!(changed.report.counts(Operation::AdjustedFit).attempted, 2);
        assert_ne!(changed.result.unwrap().identity, reused.identity);
    }
}

/// Retained typed cyclic family program, independent of report and artifact production.
#[derive(Clone, Debug)]
pub struct CyclicFamilyPreparedState {
    pub(super) program:
        std::sync::Arc<antecedent_io::cyclic_family_artifact::CompiledCyclicFamilyProgram>,
}
/// Cheap inspection of a typed cyclic family preparation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CyclicFamilyPreparedInspection {
    /// Complete target/source/semantics/numerical-policy identity.
    pub identity: String,
    /// Explicit equilibrium/dynamics/equation/graph family.
    pub family: String,
    /// Compilation performs zero statistical/numerical executions.
    pub preparation_evaluations: usize,
    /// Actual reservation held by the compiled owner.
    pub retained_bytes: u64,
    /// Existing SCC, dynamic and equation compiler products, retaining their semantics.
    pub execution_plans: Vec<antecedent_model::compile::CompiledSystemPlan>,
}
impl Study {
    /// Compile any typed cyclic family without fitting, sampling, solving or artifact encoding.
    ///
    /// # Errors
    /// Invalid model/program inputs, cancellation, or retained-resource refusal.
    pub fn prepare_cyclic_family(
        request: &antecedent_io::cyclic_family_artifact::CyclicFamilyRequest,
        ctx: &ExecutionContext,
    ) -> Result<PreparedStudy<CyclicFamilyPreparedState>, CyclicArtifactError> {
        let program = antecedent_io::cyclic_family_artifact::CompiledCyclicFamilyProgram::compile(
            request, ctx,
        )?;
        Ok(PreparedStudy {
            state: CyclicFamilyPreparedState { program: std::sync::Arc::new(program) },
        })
    }
}
impl PreparedStudy<CyclicFamilyPreparedState> {
    /// Inspect the frozen family and semantic program without executing its numerical engine.
    #[must_use]
    pub fn inspect(&self) -> CyclicFamilyPreparedInspection {
        CyclicFamilyPreparedInspection {
            identity: self.state.program.identity().into(),
            family: self.state.program.request().family_name().into(),
            preparation_evaluations: 0,
            retained_bytes: self.state.program.retained_bytes(),
            execution_plans: self.state.program.execution_plans().to_vec(),
        }
    }
    /// Execute the retained typed program; compatible downstream worlds reuse frozen fits.
    ///
    /// # Errors
    /// Failed model execution, cancellation, or resource refusal.
    pub fn estimate(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<antecedent_io::cyclic_family_artifact::CyclicFamilyReport, CyclicArtifactError>
    {
        self.state.program.execute(ctx)
    }
    /// Preserve native facts even on refusal and cancellation.
    #[must_use]
    pub fn estimate_observed(
        &self,
        ctx: &ExecutionContext,
    ) -> antecedent_core::execution_attempt::ExecutionAttempt<
        antecedent_io::cyclic_family_artifact::CyclicFamilyReport,
        CyclicArtifactError,
    > {
        antecedent_core::execution_attempt::observe_with_context(ctx, || self.estimate(ctx))
    }
    /// Recompile changed dependencies and share only unchanged statistical inputs.
    ///
    /// # Errors
    /// Invalid dependencies or model inputs, cancellation, or resource refusal.
    pub fn rebind(
        &self,
        request: &antecedent_io::cyclic_family_artifact::CyclicFamilyRequest,
        ctx: &ExecutionContext,
    ) -> Result<Self, CyclicArtifactError> {
        Ok(Self {
            state: CyclicFamilyPreparedState {
                program: std::sync::Arc::new(self.state.program.rebind(request, ctx)?),
            },
        })
    }
    /// Execute and encode as separate steps through the same retained owner.
    ///
    /// # Errors
    /// Failed execution or encoding, cancellation, or resource refusal.
    pub fn export(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<
        (antecedent_io::cyclic_family_artifact::CyclicFamilyReport, Vec<u8>),
        CyclicArtifactError,
    > {
        self.state.program.export(ctx)
    }
}
