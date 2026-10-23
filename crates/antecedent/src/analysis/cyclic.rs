// SPDX-License-Identifier: MIT OR Apache-2.0
//! Nonlinear equilibrium specialization of the existing PreparedStudy handle.
//! Model/data/inference remain frozen; only intervention worlds can retarget
//! while retaining the fitted mechanisms. New data requires a new preparation.

use std::sync::Arc;

use crate::analysis::lowering::cyclic::{lower_cyclic_family, lower_cyclic_system};
use crate::analysis::{PreparedStudy, Study};
use crate::planner::lowered::LoweredProgram;
use antecedent_core::{ContractIdentities, ExecutionContext, SemanticDigest};
use antecedent_io::contract_section::cyclic::CyclicContractWire;
use antecedent_io::cyclic_family_artifact::{CompiledCyclicFamilyProgram, CyclicFamilyRequest};
use antecedent_io::cyclic_system_artifact::{
    CyclicArtifactError, CyclicExecutionReport, CyclicExecutionRequest, PreparedCyclicExecution,
};
use antecedent_model::nonlinear_equilibrium::Clamp;

/// Frozen native state of a cyclic specialization of `PreparedStudy`.
#[derive(Clone, Debug)]
pub struct CyclicPreparedState {
    product: PreparedCyclicExecution,
    lowered: Arc<LoweredProgram>,
}

/// Inspection of a retained cyclic physical program.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CyclicPreparedInspection {
    /// Complete target/source/inference request identity.
    pub identity: String,
    /// Identity of the lowered program the handle executes; results carry it.
    pub lowered_program_identity: SemanticDigest,
    /// Fits actually performed by execution on this handle (zero before execution).
    pub fits_performed: usize,
    /// Per-world SCC solve schedules, sources before sinks.
    pub solve_blocks: Vec<Vec<Vec<usize>>>,
    /// Typed mechanism and simultaneous-equation execution plans.
    pub execution_plans: Vec<antecedent_model::compile::CompiledSystemPlan>,
}

/// Contract of a prepared cyclic handle: the layered identities, the cyclic
/// layers, and the contract section its family export writes.
///
/// The identities are derived from the request and its compiled schedules
/// without fitting or solving. A specified or fitted model stays
/// model-conditional; the contract grants no scientific license.
#[derive(Clone, Debug, PartialEq)]
pub struct CyclicFamilyContract {
    /// Target, identification, identification product, program, inference
    /// binding, observation and data snapshot.
    pub identities: ContractIdentities,
    /// `cyclic.solution.v1` tag of the solve semantics.
    pub cyclic_solution: SemanticDigest,
    /// Recovery identity (experiment data snapshot, regime contract, recovery
    /// derivation), for an experimental recovery.
    pub recovery: Option<SemanticDigest>,
    /// Identity of the lowered program the handle's results carry.
    pub lowered_program_identity: SemanticDigest,
    /// The contract section a family export writes.
    pub section: CyclicContractWire,
}

impl CyclicFamilyContract {
    /// Declare this certified cyclic product as the composition component `component`.
    ///
    /// The declaration carries the product's program identity and
    /// `cyclic.solution.v1` tag; the product's feedback stays inside that one
    /// component. A cycle among components is still refused by
    /// [`crate::analysis::check_interface_composition_with_cyclic`].
    #[must_use]
    pub fn as_component(
        &self,
        component: impl Into<String>,
    ) -> crate::analysis::CyclicComponentDeclaration {
        crate::analysis::CyclicComponentDeclaration {
            component: component.into(),
            program: SemanticDigest::from_bytes(self.section.identities.program),
            cyclic_solution: self.cyclic_solution,
        }
    }

    fn of(section: CyclicContractWire, lowered: &LoweredProgram) -> Self {
        Self {
            identities: section.contract_identities(),
            cyclic_solution: SemanticDigest::from_bytes(section.cyclic_solution),
            recovery: section.recovery.map(SemanticDigest::from_bytes),
            lowered_program_identity: lowered.identity(),
            section,
        }
    }
}

/// Context for compiling solve schedules outside a caller's context. Lowering
/// reads no seed and runs no numerical engine.
fn lowering_context() -> ExecutionContext {
    ExecutionContext::production(0, 1)
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
        let lowered = Arc::new(lower_cyclic_system(&request, ctx)?);
        let product = PreparedCyclicExecution::prepare(request, ctx)?;
        Ok(PreparedStudy { state: CyclicPreparedState { product, lowered } })
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
            lowered_program_identity: self.state.lowered.identity(),
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
        let identity = self.state.lowered.identity();
        self.state
            .product
            .estimate(ctx)
            .map(|report| report.with_lowered_program_identity(identity))
    }

    /// The lowered program this handle executes.
    #[must_use]
    pub fn lowered(&self) -> &LoweredProgram {
        &self.state.lowered
    }

    /// Contract of this handle, derived through the `system` family the request is
    /// carried as. Deriving it compiles schedules and does not fit.
    ///
    /// # Errors
    ///
    /// The request does not compile as a family, or a payload does not encode.
    pub fn contract(&self) -> Result<CyclicFamilyContract, CyclicArtifactError> {
        let family = CyclicFamilyRequest::System { request: self.request().clone() };
        let program = CompiledCyclicFamilyProgram::compile(&family, &lowering_context())?;
        Ok(CyclicFamilyContract::of(program.contract()?, &self.state.lowered))
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
        antecedent_core::execution_attempt::observe_with_context(ctx, || self.estimate(ctx))
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
        let product = self.state.product.retarget(worlds, noise, initial)?;
        let lowered = Arc::new(lower_cyclic_system(product.request(), &lowering_context())?);
        Ok(Self { state: CyclicPreparedState { product, lowered } })
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
        let identity = self.state.lowered.identity();
        let (report, bytes) = self.state.product.export(artifact_id, ctx)?;
        Ok((report.with_lowered_program_identity(identity), bytes))
    }

    /// The same frozen request as a family program, which shares rebinding,
    /// evidence projection, keyed replay and fit reuse with every other family.
    ///
    /// Compilation does not fit. The family program keeps its own fit cache: a fit
    /// already performed by this handle is not transferred.
    ///
    /// # Errors
    /// Invalid program inputs, cancellation, or retained-resource refusal.
    pub fn to_family(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<PreparedStudy<CyclicFamilyPreparedState>, CyclicArtifactError> {
        Study::prepare_cyclic_family(
            &antecedent_io::cyclic_family_artifact::CyclicFamilyRequest::System {
                request: self.request().clone(),
            },
            ctx,
        )
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

        assert_family_reuses_fit_after_rebinding(request, &ctx);
    }

    fn assert_family_reuses_fit_after_rebinding(
        request: CyclicExecutionRequest,
        ctx: &ExecutionContext,
    ) {
        use antecedent_core::execution_attempt::Operation;
        use antecedent_estimate::cyclic_likelihood::CyclicBootstrapDesign;
        use antecedent_io::cyclic_family_artifact::{
            CyclicFamilyAnswer, CyclicFamilyRequest, consume_cyclic_family,
        };
        let fit = request.fit.unwrap();
        let mut family = CyclicFamilyRequest::FrequentistBootstrap {
            design: CyclicBootstrapDesign::default(),
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
        let preparation = antecedent_core::execution_attempt::observe_with_context(ctx, || {
            Study::prepare_cyclic_family(&family, ctx)
        });
        assert_eq!(preparation.report.counts(Operation::AdjustedFit).attempted, 0);
        let compiled = preparation.result.unwrap();
        assert_eq!(compiled.inspect().preparation_evaluations, 0);
        let first = compiled.estimate_observed(ctx);
        assert_eq!(first.report.counts(Operation::AdjustedFit).attempted, 2);
        let first = first.result.unwrap();
        if let CyclicFamilyRequest::FrequentistBootstrap { worlds, .. } = &mut family {
            *worlds = vec![vec![Clamp { variable: 0, value: 2.0 }]];
        }
        let rebound = compiled.rebind(&family, ctx).unwrap();
        drop(compiled);
        let reused = rebound.estimate_observed(ctx);
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
        let (exported, bytes) = rebound.export(ctx).unwrap();
        assert_eq!(consume_cyclic_family(&bytes, &exported.identity, ctx).unwrap(), exported);
        if let CyclicFamilyRequest::FrequentistBootstrap { rows, .. } = &mut family {
            for row in rows {
                row.values[1] *= 0.5;
            }
        }
        let refreshed = rebound.rebind(&family, ctx).unwrap();
        let changed = refreshed.estimate_observed(ctx);
        assert_eq!(changed.report.counts(Operation::AdjustedFit).attempted, 2);
        assert_ne!(changed.result.unwrap().identity, reused.identity);
    }
}

/// Retained typed cyclic family program, independent of report and artifact production.
#[derive(Clone, Debug)]
pub struct CyclicFamilyPreparedState {
    pub(super) program:
        std::sync::Arc<antecedent_io::cyclic_family_artifact::CompiledCyclicFamilyProgram>,
    /// The lowered program the retained program was lowered to.
    pub(super) lowered: Arc<LoweredProgram>,
}

impl CyclicFamilyPreparedState {
    fn lower(program: CompiledCyclicFamilyProgram) -> Result<Self, CyclicArtifactError> {
        let lowered = Arc::new(lower_cyclic_family(&program)?);
        Ok(Self { program: Arc::new(program), lowered })
    }
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
    /// Identity of the lowered program the handle executes; results carry it.
    pub lowered_program_identity: SemanticDigest,
    /// Identification the lowered program carries: `specified_model_conditional`,
    /// `experimental_recovery` or `cyclic_ioscm`. Never a graphical product.
    pub identification: &'static str,
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
        let program = CompiledCyclicFamilyProgram::compile(request, ctx)?;
        Ok(PreparedStudy { state: CyclicFamilyPreparedState::lower(program)? })
    }

    /// Compile a recovered linear family. Any other family is refused.
    ///
    /// Compilation performs no fit and does not certify the draws. The recovered
    /// law stays conditional on the clamp-experiment recovery named by the request.
    ///
    /// # Errors
    /// The request is not a recovered linear family, or [`Self::prepare_cyclic_family`]
    /// refuses the program.
    pub fn prepare_recovered_linear(
        request: &antecedent_io::cyclic_family_artifact::CyclicFamilyRequest,
        ctx: &ExecutionContext,
    ) -> Result<PreparedStudy<CyclicFamilyPreparedState>, CyclicArtifactError> {
        if !matches!(
            request,
            antecedent_io::cyclic_family_artifact::CyclicFamilyRequest::RecoveredLinear { .. }
        ) {
            return Err(CyclicArtifactError::Invalid(
                "prepare_recovered_linear accepts only a recovered linear family".into(),
            ));
        }
        Self::prepare_cyclic_family(request, ctx)
    }

    /// Compile a linear law to be recovered from clamp experiments. Any other
    /// family is refused.
    ///
    /// Compilation admits the worlds on the retained units and does not fit.
    /// Execution fits the declared regimes and draws the declared uncertainty; the
    /// law stays conditional on the clamp-experiment recipe and grants no
    /// scientific license.
    ///
    /// # Errors
    /// The request is not an experimental recovery, a world names an action a
    /// recovered law cannot evaluate, or [`Self::prepare_cyclic_family`] refuses it.
    pub fn prepare_experimental_recovery(
        request: &antecedent_io::cyclic_family_artifact::CyclicFamilyRequest,
        ctx: &ExecutionContext,
    ) -> Result<PreparedStudy<CyclicFamilyPreparedState>, CyclicArtifactError> {
        if !matches!(
            request,
            antecedent_io::cyclic_family_artifact::CyclicFamilyRequest::ExperimentalRecovery { .. }
        ) {
            return Err(CyclicArtifactError::Invalid(
                "prepare_experimental_recovery accepts only an experimental recovery family; \
                 next attempt: prepare a supplied law with prepare_recovered_linear"
                    .into(),
            ));
        }
        Self::prepare_cyclic_family(request, ctx)
    }
}
impl PreparedStudy<CyclicFamilyPreparedState> {
    /// Execute and return the intervention worlds, keyed by scientific world id,
    /// with their certificates, solve routes and failed or withheld mass.
    ///
    /// # Errors
    /// A family without intervention worlds (the refusal names the typed
    /// families), failed execution, cancellation, or resource refusal.
    pub fn interventions(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<
        Vec<antecedent_io::cyclic_family_artifact::CyclicInterventionWorld>,
        CyclicArtifactError,
    > {
        let report = self.estimate(ctx)?;
        self.state.program.request().intervention_worlds(&report).ok_or_else(|| {
            CyclicArtifactError::Invalid(format!(
                "family `{}` has no intervention worlds; next attempt: declare the worlds as \
                 typed interventions with `linear_worlds`, `equilibrium_worlds`, `ode_worlds`, \
                 `recovered_linear` or `experimental_recovery`",
                self.state.program.request().family_name()
            ))
        })
    }

    /// Inspect the frozen family and semantic program without executing its numerical engine.
    #[must_use]
    pub fn inspect(&self) -> CyclicFamilyPreparedInspection {
        CyclicFamilyPreparedInspection {
            identity: self.state.program.identity().into(),
            family: self.state.program.request().family_name().into(),
            preparation_evaluations: 0,
            retained_bytes: self.state.program.retained_bytes(),
            execution_plans: self.state.program.execution_plans().to_vec(),
            lowered_program_identity: self.state.lowered.identity(),
            identification: self
                .state
                .lowered
                .non_graphical_identification()
                .map_or("specified_model_conditional", |identification| identification.as_str()),
        }
    }

    /// The lowered program this handle executes.
    #[must_use]
    pub fn lowered(&self) -> &LoweredProgram {
        &self.state.lowered
    }

    /// Contract of this handle: layered identities over payloads derived from the
    /// request and its compiled schedules, the `cyclic.solution.v1` tag, the
    /// recovery layer of an experimental recovery, and the contract section the
    /// export writes. Deriving it does not fit, sample or solve.
    ///
    /// # Errors
    ///
    /// A payload that does not encode.
    pub fn contract(&self) -> Result<CyclicFamilyContract, CyclicArtifactError> {
        Ok(CyclicFamilyContract::of(self.state.program.contract()?, &self.state.lowered))
    }

    /// Coefficient rows the clamp design of an experimental recovery leaves
    /// unresolved, known before any fit: row `target` is unresolved when some
    /// source in `missing_clamps` is never clamped while `target` is passive.
    ///
    /// `None` for every other family. Reading it does not fit, sample or solve. A
    /// fit can leave further directions unresolved (a deficient constraint rank);
    /// those are reported by the recovered law of an execution.
    #[must_use]
    pub fn design_unresolved(
        &self,
    ) -> Option<Vec<antecedent_io::contract_section::cyclic::CyclicUnresolvedWire>> {
        match self.state.program.request() {
            CyclicFamilyRequest::ExperimentalRecovery { regimes, dropped, .. } => {
                Some(antecedent_io::cyclic_family_artifact::design_unresolved(regimes, dropped))
            }
            _ => None,
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
        let identity = self.state.lowered.identity();
        self.state.program.execute(ctx).map(|report| report.with_lowered_program_identity(identity))
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
            state: CyclicFamilyPreparedState::lower(self.state.program.rebind(request, ctx)?)?,
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
        let identity = self.state.lowered.identity();
        let (report, bytes) = self.state.program.export(ctx)?;
        Ok((report.with_lowered_program_identity(identity), bytes))
    }
}

#[cfg(test)]
mod recovered_linear {
    use super::*;
    use antecedent_core::execution_attempt::Operation;
    use antecedent_io::cyclic_family_artifact::{
        CyclicFamilyAnswer, CyclicFamilyRequest, RecoveredCoefficientWire, RecoveredDrawAnswer,
        RecoveredDrawWire, RecoveredInterventionWire, RecoveredSolvedBy, consume_cyclic_family,
    };
    use antecedent_model::compile::{ExecutionBlock, SystemSemantics};

    fn draw(into_from: (f64, f64), weight: f64, marked: bool) -> RecoveredDrawWire {
        RecoveredDrawWire {
            coefficients: vec![
                RecoveredCoefficientWire { into: 0, from: 1, value: into_from.0 },
                RecoveredCoefficientWire { into: 1, from: 0, value: into_from.1 },
            ],
            intercept: vec![0.0, 0.0],
            noise_mean: vec![1.0, 0.0],
            noise_covariance: vec![1.0, 0.0, 0.0, 1.0],
            unresolved: Vec::new(),
            weight,
            marked_inadmissible: marked,
        }
    }

    #[test]
    fn recovered_linear_prepares_without_a_fit_and_keeps_failed_draws() {
        let request = CyclicFamilyRequest::RecoveredLinear {
            nodes: vec![0, 1],
            draws: vec![
                draw((0.4, 0.5), 0.5, false),
                draw((0.4, 0.5), 0.25, true),
                draw((1.0, 1.0), 0.25, false),
            ],
            worlds: vec![
                RecoveredInterventionWire { clamp_nodes: vec![], clamp_values: vec![] },
                RecoveredInterventionWire { clamp_nodes: vec![0], clamp_values: vec![2.0] },
            ],
            outcome: 1,
            source_id: "clamp-experiments".into(),
            certifying_evidence: vec!["held-out-regime".into()],
            uncertainty: antecedent_io::cyclic_family_artifact::RecoveredDrawOriginWire::Supplied,
        };
        let ctx = ExecutionContext::for_tests(1);
        let prepared = observe(&ctx, || Study::prepare_cyclic_family(&request, &ctx));
        assert_eq!(prepared.report.counts(Operation::AdjustedFit).attempted, 0);
        let prepared = prepared.result.unwrap();
        let inspection = prepared.inspect();
        assert_eq!(inspection.preparation_evaluations, 0);
        assert_eq!(inspection.family, "recovered_linear");
        assert_eq!(inspection.execution_plans[0].semantics, SystemSemantics::EquilibriumScm);
        assert!(
            inspection.execution_plans[0]
                .blocks
                .iter()
                .any(|block| { matches!(block, ExecutionBlock::SimultaneousEquations { .. }) })
        );
        assert_eq!(inspection.execution_plans[1].semantics, SystemSemantics::AcyclicScm);
        assert!(
            inspection.execution_plans[1]
                .blocks
                .iter()
                .all(|block| { matches!(block, ExecutionBlock::Mechanism { .. }) })
        );
        let executed = prepared.estimate_observed(&ctx);
        assert_eq!(executed.report.counts(Operation::AdjustedFit).attempted, 0);
        let report = executed.result.unwrap();
        let CyclicFamilyAnswer::RecoveredLinear(answer) = &report.answer else {
            panic!("recovered linear answer");
        };
        let open = &answer.worlds[0].draws;
        let clamped = &answer.worlds[1].draws;
        match &open[0].outcome {
            RecoveredDrawAnswer::Equilibrium {
                solved_by: RecoveredSolvedBy::Cyclic { next_attempt },
                mean,
                ..
            } => {
                assert!(next_attempt.contains("condensation"), "{next_attempt}");
                assert!((mean[0] - 1.25).abs() < 1e-8);
                assert!((mean[1] - 0.625).abs() < 1e-8);
            }
            other => panic!("open draw was {other:?}"),
        }
        match &clamped[0].outcome {
            RecoveredDrawAnswer::Equilibrium {
                solved_by: RecoveredSolvedBy::Acyclic,
                mean,
                ..
            } => {
                assert!((mean[0] - 2.0).abs() < 1e-8);
                assert!((mean[1] - 1.0).abs() < 1e-8);
            }
            other => panic!("clamped draw was {other:?}"),
        }
        assert!(matches!(
            &open[1].outcome,
            RecoveredDrawAnswer::Inadmissible { reason } if reason == "marked"
        ));
        assert!(matches!(&open[2].outcome, RecoveredDrawAnswer::NotCertified { .. }));
        assert_eq!(open.len(), 3);
        assert!((answer.worlds[0].withheld_mass - 0.5).abs() < 1e-12);
        let (exported, bytes) = prepared.export(&ctx).unwrap();
        assert_eq!(consume_cyclic_family(&bytes, &exported.identity, &ctx).unwrap(), exported);
        assert_eq!(prepared.inspect().preparation_evaluations, 0);
    }

    fn observe<T, E>(
        ctx: &ExecutionContext,
        body: impl FnOnce() -> Result<T, E>,
    ) -> antecedent_core::execution_attempt::ExecutionAttempt<T, E> {
        antecedent_core::execution_attempt::observe_with_context(ctx, body)
    }
}
