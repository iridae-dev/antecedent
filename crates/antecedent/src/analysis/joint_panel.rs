//! Checked execution of the reviewed nonlinear multi-channel panel mechanism.
//!
//! The common prepared handle owns the scientific question, real design attempts,
//! statistical program, physical choice and dependency keys. Numerical fitting remains
//! in antecedent-estimate. A finite fitted response is never an identification proof.
// SPDX-License-Identifier: MIT OR Apache-2.0

mod artifact;
mod categorical;
pub use categorical::{
    DummyCodedPanel, PanelCategoricalChannel, PanelDummyChannel, dummy_code_panel,
};
mod contracts;
mod data_policy;
pub use data_policy::{
    PanelCellRole, PanelDataAccounting, PanelMissingPolicy, PanelUnitAccounting, PanelUnitDrop,
};
/// Native design-specific joint functionals and retained Bayesian laws.
pub mod designs;
pub use designs::{
    DesignArtifactLimits, JointDesignProof, JointDesignResult, JointDesignWorldBinding,
    PreparedJointDesign, replay_joint_design_artifact, replay_joint_design_artifact_with_limits,
};
mod experiment;
mod explanation;
pub use explanation::{
    ChangeComponent, CheckedReferenceDecomposition, DecompositionEndpoint,
    DecompositionJointUncertainty, DesignAdaptiveReference, HistoricalPopulationReference,
    HistoricalReferenceDecomposition, HybridReferenceSupport, IdentifiedComponentStep,
    IdentifiedReferenceAttempt, JointPanelExplanationAttempt, JointPanelExplanationRequest,
    MechanismPeriodMap, NextDesignStep, PanelDecomposition, PanelDecompositionRequest,
    PopulationReferenceAttempt, PopulationReferenceBindings, PopulationReferenceDecomposition,
    PopulationReferenceTransportProof, ReferenceArtifactLimits, ReferenceComponentBindings,
    ReferenceDecomposition, ReferenceDecompositionCovariance, ReferenceObligation,
    ScientificReferenceDecomposition, StreamedDecomposition, decompose_population_reference,
    decompose_scientific_reference, replay_population_reference,
    replay_population_reference_with_limits, replay_reference_decomposition,
    replay_reference_decomposition_with_limits,
};
mod imputed;
pub use imputed::{JointPanelImputedResult, PreparedJointPanelImputation};
mod ports;
mod projection;
mod search;
/// Held-out observational selection through actual whole-program randomized designs.
pub mod selection;
mod source;
mod static_source;
mod streamed;
pub use experiment::{
    HistoricalJointLawExperiment, HistoricalNativeJointExperiment, JointExperimentAcceptanceMatrix,
    JointExperimentAcceptanceRow, JointLawExperimentRequest, JointLawExperimentResult,
    LawExperimentLimits, LocalRatioGaussianRegularization, NativeExperimentBinding,
    NativeExperimentDecision, NativeJointExperimentResult, PanelExperimentDecisionSource,
    PanelExperimentDecisionState, PanelExperimentEvidence, PanelExperimentMapping,
    PanelExperimentModularPolicy, PanelExperimentResult, PanelExperimentState,
    PanelExperimentUpdate, PanelFutureRegime, PanelFutureSample, PanelFutureSampleValue,
    PanelFutureStudy, PanelFutureStudyUpdate, PanelFutureStudyValue,
    joint_experiment_acceptance_matrix,
};
pub use streamed::{DEFAULT_TILE_DRAWS, JointPanelProjectedLaw};
#[cfg(test)]
mod tests;

use crate::{PreparedStudy, Study};
use antecedent_core::physical_choice::PhysicalChoice;
use antecedent_core::{
    ExecutionContext, ScientificTarget, ScientificWorldId, SemanticDigest, VariableId,
};
use antecedent_data::PanelData;
use antecedent_estimate::joint_panel::time_varying::{TemporalPanelFit, TemporalPanelSpec};
use antecedent_estimate::joint_panel::{
    self as engine, PanelFit, PanelFitOptions, PanelProgram, PanelResponse,
};
use antecedent_identify::CompiledIdentificationProduct;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};

pub use ports::{
    JointPanelContrast, JointPanelContrastUncertainty, JointPanelResponseFunctional,
    JointPanelScientificScope,
};
pub use search::{
    JointPanelCertifiedSearch, JointPanelCertifiedSearchResult, JointPanelContinuousSearch,
    JointPanelContinuousSearchRequest, JointPanelContinuousSearchResult,
    JointPanelProgramSearchResult, JointPanelSearch, JointPanelSearchRequest,
    JointPanelSearchUncertainty, replay_panel_certificate,
};

pub use contracts::{
    JointPanelDesignAttempt, JointPanelDesignOutcome, JointPanelDesignProof,
    JointPanelDesignTheorem, JointPanelFunctional, JointPanelWorldSupport,
    declare_joint_panel_exogenous_context, joint_panel_target,
};

/// Source schema roles; none is inferred from column order or display labels.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JointPanelColumns {
    /// Ordered explicit observed channel assignments; engine IDs derive from these stable schema IDs.
    pub channels: Vec<VariableId>,
    /// Observed response in its declared source units.
    pub outcome: VariableId,
    /// Declared observable confounder.
    pub confounder: VariableId,
    /// Integer-coded seasonal level, with zero as the reference.
    pub season: VariableId,
}

/// One exact scientific world and its prespecified multi-channel program.
#[derive(Clone, Debug, PartialEq)]
pub struct JointPanelWorld {
    /// World key used by the common scientific target.
    pub world: ScientificWorldId,
    /// Explicit reference-world designation.
    pub reference: bool,
    /// Schedule and all group-specific future histories/covariates.
    pub program: PanelProgram,
}

impl JointPanelWorld {
    /// Number of explicitly supplied periods; compilation validates all channel axes agree.
    #[must_use]
    pub fn horizon(&self) -> usize {
        self.program.channels.first().map_or(0, |channel| channel.values.len())
    }
}

/// Complete checked request; replacing programs alone can reuse a compatible fit.
#[derive(Clone, Debug)]
pub struct JointPanelRequest {
    /// Exact source columns.
    pub columns: JointPanelColumns,
    /// Prespecified sparse interaction subsets; fitting never invents omitted terms.
    pub interactions: Vec<engine::PanelInteractionSpec>,
    /// Finite seasonal level count.
    pub seasons: usize,
    /// Zero for the static reduction, or two for the reviewed carryover family.
    pub lag_order: usize,
    /// Complete numerical policy and uncertainty realization.
    pub options: PanelFitOptions,
    /// Selected mechanism family; the default is the reviewed family. A broader family
    /// changes the numerical mechanism only and never the identification requirements.
    pub family: engine::PanelFamily,
    /// Ordered evaluation worlds, including an explicit reference when contrasted.
    pub worlds: Vec<JointPanelWorld>,
    /// Scientific target, checked against every program before fitting.
    pub target: ScientificTarget,
    /// Optional known whole-program randomized assignment at the geographic origin.
    /// Per-period randomization does not establish this design.
    pub assignment: Option<antecedent_core::RandomizedEffectQuery>,
    /// Optional reviewed piecewise/smooth time-varying mechanism attempt.
    pub temporal: Option<TemporalPanelSpec>,
}

/// Typed compiler, authority, numerical, resource and cancellation refusals.
#[derive(Clone, Debug, thiserror::Error)]
pub enum JointPanelError {
    /// Source roles, programs, history or scientific target disagree.
    #[error("joint panel compilation refused: {0}")]
    Compile(String),
    /// An identified causal consumer was requested without an earned design proof.
    #[error("joint panel causal claim withheld: {0}")]
    Unsupported(String),
    /// The retained program's engine refused execution.
    #[error(transparent)]
    Engine(#[from] engine::PanelError),
}
impl JointPanelError {
    fn compile(error: impl std::fmt::Display) -> Self {
        Self::Compile(error.to_string())
    }
}

/// Fitted mechanism retained across compatible program and utility changes.
#[derive(Clone, Debug)]
pub enum JointPanelFit {
    /// Invariant mean mechanism with stationary AR(1) group dependence.
    Invariant(Arc<PanelFit>),
    /// Checked piecewise/smooth mean mechanism selection procedure.
    Varying(Arc<TemporalPanelFit>),
}

#[derive(Debug, Default)]
struct FitCache {
    slot: Mutex<Option<(SemanticDigest, JointPanelFit)>>,
    performed: AtomicUsize,
}

/// Native retained state specializing the common prepared execution owner.
#[derive(Clone, Debug)]
pub struct JointPanelState {
    source: PanelData,
    data: Arc<engine::PanelData>,
    request: Arc<JointPanelRequest>,
    source_identity: SemanticDigest,
    options_identity: SemanticDigest,
    identification: Arc<CompiledIdentificationProduct>,
    attempts: Arc<[JointPanelDesignAttempt]>,
    physical: PhysicalChoice,
    static_source: Option<Arc<crate::PreparedJointResponse>>,
    fits: Arc<FitCache>,
    source_retention: Arc<antecedent_core::execution::AdmissionLease>,
    program_retention: Option<Arc<antecedent_core::execution::AdmissionLease>>,
    accounting: Arc<PanelDataAccounting>,
}

/// Cheap inspection; no likelihood fit, draw or response simulation occurs here.
#[derive(Clone, Debug)]
pub struct JointPanelInspection {
    /// Exact statistical source including group partition, cadence and observed prefix.
    pub source_identity: SemanticDigest,
    /// Source-independent numerical policy identity.
    pub options_identity: SemanticDigest,
    /// Complete scientific question.
    pub target: ScientificTarget,
    /// Retained per-world and assembled design obligations.
    pub identification: CompiledIdentificationProduct,
    /// Ordered actual design attempts, including the experimental fallback.
    pub design_attempts: Vec<JointPanelDesignAttempt>,
    /// Only executable physical alternatives are advertised.
    pub physical: PhysicalChoice,
    /// Fits actually run by this cache, zero immediately after compilation.
    pub fits_performed: usize,
}

/// Aligned fitted mean responses; outcome innovations are integrated out.
#[derive(Clone, Debug)]
pub struct JointPanelResult {
    /// One point response for every declared program in request order.
    pub point: Vec<PanelResponse>,
    /// One complete shared-parameter response row per requested fit draw; failures stay None.
    pub draws: Vec<Option<Vec<PanelResponse>>>,
    /// Retained numerical fit and its original convergence/uncertainty diagnostics.
    pub fit: JointPanelFit,
    program: PreparedStudy<JointPanelState>,
    dependency: SemanticDigest,
    seed: u64,
    context: ExecutionContext,
    _retention: Arc<antecedent_core::execution::AdmissionLease>,
}

impl Study {
    /// Compile the existing exact panel family into the common prepared owner.
    ///
    /// L=2 consumes the first two actual rows of each source unit as pre-period
    /// exposure history. It never fills history, cadence or future covariates implicitly.
    ///
    /// # Errors
    /// Schema, cadence, history, target, resources or explicit numerical policy are refused.
    pub fn prepare_joint_panel(
        data: PanelData,
        request: JointPanelRequest,
        ctx: &ExecutionContext,
    ) -> Result<PreparedStudy<JointPanelState>, JointPanelError> {
        PreparedStudy::<JointPanelState>::compile_joint_panel(data, request, ctx)
    }
    /// [`Study::prepare_joint_panel`] with a declared policy for missing cells.
    ///
    /// Unit sampling weights and analysis masks on the source series are read as declared;
    /// the policy decides only what a missing cell (`NaN` or a cleared validity bit) means:
    /// refuse (the default of the plain entry point), drop the unit, or exclude the affected
    /// periods with the serial law marginalized exactly over them. The exact bookkeeping is
    /// available from the prepared handle.
    ///
    /// # Errors
    /// As [`Study::prepare_joint_panel`], plus a weighted source under a bootstrap, posterior,
    /// spatial or time-varying declaration, which the weighted pseudo-likelihood cannot honour.
    pub fn prepare_joint_panel_with_policy(
        data: PanelData,
        request: JointPanelRequest,
        missing: PanelMissingPolicy,
        ctx: &ExecutionContext,
    ) -> Result<PreparedStudy<JointPanelState>, JointPanelError> {
        PreparedStudy::<JointPanelState>::compile_joint_panel_with_policy(
            data, request, missing, ctx,
        )
    }
}
impl PreparedStudy<JointPanelState> {
    /// Compile without fitting or constructing another public study builder.
    ///
    /// # Errors
    /// Invalid source, target, program, policy, cancellation or admission refusal.
    pub fn compile_joint_panel(
        source: PanelData,
        request: JointPanelRequest,
        ctx: &ExecutionContext,
    ) -> Result<Self, JointPanelError> {
        Self::compile_joint_panel_with_policy(source, request, PanelMissingPolicy::Refuse, ctx)
    }
    /// [`Self::compile_joint_panel`] with a declared policy for missing cells.
    ///
    /// # Errors
    /// Invalid source, target, program, policy, cancellation or admission refusal.
    pub fn compile_joint_panel_with_policy(
        source: PanelData,
        request: JointPanelRequest,
        missing: PanelMissingPolicy,
        ctx: &ExecutionContext,
    ) -> Result<Self, JointPanelError> {
        source::check_cancelled(ctx)?;
        let bytes = source::source_quote(&source, &request)?;
        source::check_memory(ctx, bytes)?;
        let retention = Arc::new(
            ctx.acquire_admission(0, bytes)
                .map_err(|error| engine::PanelError::Resource(error.to_string()))?,
        );
        let (data, identity, accounting) = source::adapt(&source, &request, missing, ctx)?;
        Self::lower(
            source,
            Arc::new(data),
            identity,
            Arc::new(accounting),
            request,
            Arc::new(FitCache::default()),
            retention,
            ctx,
        )
    }
    #[expect(clippy::too_many_arguments, reason = "the retained compiler product is assembled once")]
    fn lower(
        source: PanelData,
        data: Arc<engine::PanelData>,
        source_identity: SemanticDigest,
        accounting: Arc<PanelDataAccounting>,
        request: JointPanelRequest,
        fits: Arc<FitCache>,
        source_retention: Arc<antecedent_core::execution::AdmissionLease>,
        ctx: &ExecutionContext,
    ) -> Result<Self, JointPanelError> {
        source::check_cancelled(ctx)?;
        engine::validate_fit_options(&data, &request.options)?;
        let options_identity = request.options.identity(&data)?;
        let programs = request.worlds.iter().map(|world| world.program.clone()).collect::<Vec<_>>();
        engine::validate_panel_programs(&data, &programs)?;
        if let Some(spec) = &request.temporal {
            if data.is_weighted() || data.has_excluded_periods() {
                return Err(engine::PanelError::Invalid(
                    "a time-varying mechanism is selected and bootstrapped on complete, equally weighted series; sampling weights and excluded periods are not carried through its selection procedure, so they are refused rather than ignored".into(),
                )
                .into());
            }
            engine::time_varying::validate_temporal_spec(&data, spec)?;
        }
        contracts::validate_target(source.schema(), &request)?;
        let (identification, attempts) = contracts::compile_support(&data, &request)?;
        let physical = source::physical_choice(&data)?;
        Ok(Self {
            state: JointPanelState {
                source,
                data,
                request: Arc::new(request),
                source_identity,
                options_identity,
                identification: Arc::new(identification),
                attempts: attempts.into(),
                physical,
                fits,
                static_source: None,
                source_retention,
                program_retention: None,
                accounting,
            },
        })
    }
    /// Frozen checked request.
    #[must_use]
    pub fn request(&self) -> &JointPanelRequest {
        &self.state.request
    }
    /// Original panel and its schema/partition metadata.
    #[must_use]
    pub fn source_data(&self) -> &PanelData {
        &self.state.source
    }
    /// Exact bookkeeping of the units, rows and periods the source lost to analysis masks and
    /// missing cells, and whether the units carry sampling weights.
    #[must_use]
    pub fn data_accounting(&self) -> &PanelDataAccounting {
        &self.state.accounting
    }
    /// Exact adapted statistical data used by the numerical engine.
    #[must_use]
    pub fn model_data(&self) -> &engine::PanelData {
        &self.state.data
    }
    /// Fitting calls shared by compatible prepared handles.
    #[must_use]
    pub fn fits_performed(&self) -> usize {
        self.state.fits.performed.load(Ordering::SeqCst)
    }
    /// Inspect the same retained product that execution consumes.
    #[must_use]
    pub fn inspect(&self) -> JointPanelInspection {
        JointPanelInspection {
            source_identity: self.state.source_identity,
            options_identity: self.state.options_identity,
            target: self.state.request.target.clone(),
            identification: (*self.state.identification).clone(),
            design_attempts: self.state.attempts.to_vec(),
            physical: self.state.physical,
            fits_performed: self.fits_performed(),
        }
    }
    /// Change only program/world/functional bindings, retaining compatible fitted mechanisms.
    ///
    /// # Errors
    /// Program, target, design support or caller resource checks fail before fitting.
    pub fn retarget(
        &self,
        worlds: Vec<JointPanelWorld>,
        target: ScientificTarget,
        ctx: &ExecutionContext,
    ) -> Result<Self, JointPanelError> {
        source::check_cancelled(ctx)?;
        let bytes = source::retarget_quote(self.request(), &worlds, &target)?;
        source::check_memory(ctx, bytes)?;
        let retention = Arc::new(
            ctx.acquire_admission(0, bytes)
                .map_err(|error| engine::PanelError::Resource(error.to_string()))?,
        );
        let original = self.request();
        let request = JointPanelRequest {
            columns: original.columns.clone(),
            interactions: original.interactions.clone(),
            seasons: original.seasons,
            lag_order: original.lag_order,
            options: original.options.clone(),
            family: original.family.clone(),
            worlds,
            target,
            assignment: original.assignment.clone(),
            temporal: original.temporal.clone(),
        };
        let mut next = Self::lower(
            self.state.source.clone(),
            Arc::clone(&self.state.data),
            self.state.source_identity,
            Arc::clone(&self.state.accounting),
            request,
            Arc::clone(&self.state.fits),
            Arc::clone(&self.state.source_retention),
            ctx,
        )?;
        next.state.program_retention = Some(retention);
        Ok(next)
    }
    /// Replace actual fitting dependencies; cached fits are reused only for an identical key.
    ///
    /// # Errors
    /// New source or inference options fail compilation, or resources/cancellation refuse it.
    pub fn refit_data(
        &self,
        data: PanelData,
        options: PanelFitOptions,
        ctx: &ExecutionContext,
    ) -> Result<Self, JointPanelError> {
        let mut request = (*self.state.request).clone();
        request.options = options;
        let mut prepared = Self::compile_joint_panel_with_policy(
            data,
            request,
            self.state.accounting.policy,
            ctx,
        )?;
        prepared.state.fits = Arc::clone(&self.state.fits);
        Ok(prepared)
    }
    /// Execute from the retained compiler product with shared admission and cancellation.
    ///
    /// # Errors
    /// Numerical, dependency, cancellation and resource errors remain typed.
    pub fn estimate(&self, ctx: &ExecutionContext) -> Result<JointPanelResult, JointPanelError> {
        let mut result =
            ctx.try_map_indexed(1, |_, worker| self.estimate_impl(worker)).map_err(|error| {
                match error {
                    antecedent_core::execution::MapExecutionError::Work(error) => error,
                    antecedent_core::execution::MapExecutionError::Admission(
                        antecedent_core::execution::AdmissionError::Cancelled,
                    ) => engine::PanelError::Cancelled.into(),
                    antecedent_core::execution::MapExecutionError::Admission(error) => {
                        engine::PanelError::Resource(error.to_string()).into()
                    }
                }
            })?;
        result
            .pop()
            .ok_or_else(|| JointPanelError::compile("admitted panel worker returned no result"))
    }
    fn estimate_impl(&self, ctx: &ExecutionContext) -> Result<JointPanelResult, JointPanelError> {
        source::check_cancelled(ctx)?;
        self.state
            .identification
            .validate_target(&self.state.request.target)
            .map_err(JointPanelError::compile)?;
        self.state.source_retention.observe_current();
        if let Some(lease) = &self.state.program_retention {
            lease.observe_current();
        }
        let dependency = source::fit_dependency(&self.state, ctx)?;
        let fit = self.fit(dependency, ctx)?;
        self.evaluate_fit(fit, dependency, ctx)
    }
    fn evaluate_fit(
        &self,
        fit: JointPanelFit,
        dependency: SemanticDigest,
        ctx: &ExecutionContext,
    ) -> Result<JointPanelResult, JointPanelError> {
        source::check_cancelled(ctx)?;
        self.state.source_retention.observe_current();
        if let Some(lease) = &self.state.program_retention {
            lease.observe_current();
        }
        self.state
            .identification
            .validate_target(&self.state.request.target)
            .map_err(JointPanelError::compile)?;
        let programs =
            self.state.request.worlds.iter().map(|world| world.program.clone()).collect::<Vec<_>>();
        let responses = match &fit {
            JointPanelFit::Invariant(fit) => {
                let _ = fit.retained_bytes();
                engine::evaluate_panel_law(
                    &self.state.data,
                    &fit.point,
                    &fit.draws,
                    &programs,
                    ctx,
                )?
            }
            JointPanelFit::Varying(fit) => {
                let _ = fit.retained_bytes();
                engine::time_varying::evaluate_temporal_law(
                    &self.state.data,
                    &fit.point,
                    &fit.draws,
                    &programs,
                    ctx,
                )?
            }
        };
        let (point, draws, retention) = responses.into_parts();
        Ok(JointPanelResult {
            point,
            draws,
            fit,
            program: self.clone(),
            dependency,
            seed: ctx.rng.master_seed(),
            context: ctx.clone(),
            _retention: retention,
        })
    }
    fn fit(
        &self,
        dependency: SemanticDigest,
        ctx: &ExecutionContext,
    ) -> Result<JointPanelFit, JointPanelError> {
        let mut cache = loop {
            source::check_cancelled(ctx)?;
            match self.state.fits.slot.try_lock() {
                Ok(cache) => break cache,
                Err(std::sync::TryLockError::Poisoned(_)) => {
                    return Err(JointPanelError::Compile("fit cache was poisoned".into()));
                }
                Err(std::sync::TryLockError::WouldBlock) => {
                    std::thread::park_timeout(std::time::Duration::from_millis(10))
                }
            }
        };
        if ctx.cache_policy.enabled {
            if let Some((key, fit)) = cache.as_ref() {
                if *key == dependency {
                    source::observe_reuse();
                    return Ok(fit.clone());
                }
            }
        }
        *cache = None;
        source::check_cancelled(ctx)?;
        let fit = match &self.state.request.temporal {
            Some(spec) => {
                JointPanelFit::Varying(Arc::new(engine::time_varying::fit_temporal_panel(
                    &self.state.data,
                    spec,
                    &self.state.request.options,
                    ctx,
                )?))
            }
            None => JointPanelFit::Invariant(Arc::new(engine::fit_panel(
                &self.state.data,
                &self.state.request.options,
                ctx,
            )?)),
        };
        self.state.fits.performed.fetch_add(1, Ordering::SeqCst);
        let bytes = match &fit {
            JointPanelFit::Invariant(fit) => fit.retained_bytes(),
            JointPanelFit::Varying(fit) => fit.retained_bytes(),
        };
        if ctx.cache_policy.enabled && ctx.cache_policy.max_bytes.is_none_or(|limit| bytes <= limit)
        {
            *cache = Some((dependency, fit.clone()));
        }
        Ok(fit)
    }
    /// Native facts survive success, resource refusal and cancellation.
    #[must_use]
    pub fn estimate_observed(
        &self,
        ctx: &ExecutionContext,
    ) -> antecedent_core::execution_attempt::ExecutionAttempt<JointPanelResult, JointPanelError>
    {
        antecedent_core::execution_attempt::observe_with_context(ctx, || self.estimate(ctx))
    }
}
impl JointPanelResult {
    /// Evaluate compatible worlds using this exact retained fit and its aligned uncertainty rows.
    /// Fitting dependencies remain frozen; caller context controls only projection resources/cancellation.
    ///
    /// # Errors
    /// Changed scientific/program bindings, cancellation, resource or projection failures.
    pub fn retarget_frozen(
        &self,
        worlds: Vec<JointPanelWorld>,
        target: ScientificTarget,
        ctx: &ExecutionContext,
    ) -> Result<Self, JointPanelError> {
        let prepared = self.program.retarget(worlds, target, ctx)?;
        let mut values = ctx
            .try_map_indexed(1, |_, worker| {
                prepared.evaluate_fit(self.fit.clone(), self.dependency, worker)
            })
            .map_err(|error| match error {
                antecedent_core::execution::MapExecutionError::Work(error) => error,
                antecedent_core::execution::MapExecutionError::Admission(
                    antecedent_core::execution::AdmissionError::Cancelled,
                ) => engine::PanelError::Cancelled.into(),
                antecedent_core::execution::MapExecutionError::Admission(error) => {
                    engine::PanelError::Resource(error.to_string()).into()
                }
            })?;
        let mut result = values
            .pop()
            .ok_or_else(|| JointPanelError::compile("frozen projection returned no result"))?;
        result.seed = self.seed;
        Ok(result)
    }

    /// Exact checked request and response coordinate order.
    #[must_use]
    pub fn request(&self) -> &JointPanelRequest {
        self.program.request()
    }
    /// Common target-bound identification/support product.
    pub fn identification(&self) -> &CompiledIdentificationProduct {
        &self.program.state.identification
    }
    /// Actual ordered design attempts, not a fit-derived permission label.
    #[must_use]
    pub fn design_attempts(&self) -> &[JointPanelDesignAttempt] {
        &self.program.state.attempts
    }
    /// Source, fit policy and actual execution seed/kernel dependency identity.
    #[must_use]
    pub fn dependency_identity(&self) -> SemanticDigest {
        self.dependency
    }
    /// Borrow the retained common prepared execution owner.
    #[must_use]
    pub fn prepared(&self) -> &PreparedStudy<JointPanelState> {
        &self.program
    }
    /// Refuse causal authorization when no required design earned the assembled target.
    ///
    /// # Errors
    /// Unsupported worlds and missing design premises remain explicit.
    pub fn require_identified(&self) -> Result<(), JointPanelError> {
        if self.identification().is_identified_claim() {
            Ok(())
        } else {
            Err(JointPanelError::Unsupported(
                "the model fit does not discharge the retained world/design obligations".into(),
            ))
        }
    }
}
