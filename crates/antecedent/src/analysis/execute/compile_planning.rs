//! Compile that starts from an experiment-planning request.
//!
//! [`compile_planning`] lowers a [`PlanningRequest`] for a future trial that
//! binds no data: the scientific problem is the assignment contrast of the
//! planned design keyed by the request's identity, identification is by the
//! planned randomization, the statistical program names the analysis the
//! design will be read with (a sampling realization for power, a conjugate
//! posterior for a Bayesian criterion), and layer four carries the request
//! with every planning assumption. [`Study::compile_for_planning`] attaches the
//! same request to the lowered program of a randomized study whose
//! randomization is the planned design, so the plan is for the next trial of
//! that analysis.
//!
//! Compile refuses an invalid request, a design whose operating
//! characteristics cannot be established (naming the next attempt), and a
//! simulation whose replicate store the admission pool cannot hold. Nothing is
//! solved or simulated at compile. [`PlanningPlan::execute`] solves the
//! analytic plan, quotes the simulation at the planned size, leases it from
//! the admission pool before allocating, and runs it.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_contract::DataClassification;
use antecedent_core::{
    ExecutionContext, LikelihoodModel, NextAttempt, OrdinaryBayesianLikelihood, Prior,
    RandomizationDesign, ScoreFrequentist, ScoreModel, SemanticDigest,
};
use antecedent_design::planning::{
    PlannedDesign, PlanningGoal, PlanningRefusal, PlanningRequest, PlanningResult, plan_analytic,
};

use super::Study;
use super::planning_simulation::{scratch_bytes, simulate};
use super::route_input::RouteInput;
use crate::analysis::lowering::{CompileGoal, refuse_reentry};
use crate::error::CausalError;
use crate::planner::lowered::{
    ClaimDemand, Composition, DesignShape, EvidenceDecisionProgram, EvidenceRole,
    EvidenceSourceRef, FunctionalKind, IdentificationLayer, LoweredProgram, PhysicalLayer,
    PlanningLoweredParts, PlanningProgram, ProblemTarget, ScientificProblem, SimulationQuote,
    UncertaintyReading,
};
use crate::planner::statistical::{
    ModelRole, ModelSlot, PosteriorAlgorithm, PosteriorPlan, Realization, StatisticalProgram,
};

/// Why a planning program retains no identification product.
const PLANNING_IDENTIFIES_BY_DESIGN: &str = "a planned trial binds no data; its randomization \
                                              identifies the assignment contrast when it is \
                                              analysed";
/// Why a planning program's design shape is not a quoted adjustment design.
const PLANNING_DESIGN_SHAPE: &str =
    "a planning program sizes a future trial; it fits no adjustment design";

/// Plan without a study, or declare the study's randomization as the planned design.
pub const STANDALONE_PLANNING: NextAttempt = NextAttempt::new(
    "standalone_planning",
    "compile the request with compile_planning, which plans a future trial without a study, or \
     compile a randomized study whose declared randomization is the planned design",
);

/// A compiled planning request and its lowered program.
#[derive(Clone, Debug)]
pub struct PlanningPlan {
    lowered: Arc<LoweredProgram>,
    program: PlanningProgram,
}

/// An executed plan: the analytic plan, its simulation when requested, and
/// the identity of the program it executed.
#[derive(Clone, Debug, PartialEq)]
pub struct PlannedExperiment {
    /// Identity of the lowered program the plan executed.
    pub program_identity: SemanticDigest,
    /// The plan and its operating characteristics.
    pub result: PlanningResult,
}

impl PlanningPlan {
    /// The lowered program.
    #[must_use]
    pub fn lowered(&self) -> &LoweredProgram {
        &self.lowered
    }

    /// One view over every layer of the planning program. Reading it simulates nothing.
    #[must_use]
    pub fn plan_view(&self) -> crate::planner::PlanView {
        crate::planner::PlanView::of_program(
            Arc::clone(&self.lowered),
            crate::planner::PlanStage::Compiled,
        )
    }

    /// Identity of the lowered program.
    #[must_use]
    pub fn identity(&self) -> SemanticDigest {
        self.lowered.identity()
    }

    /// The declared request.
    #[must_use]
    pub fn request(&self) -> &PlanningRequest {
        self.program.request()
    }

    /// Simulation store quoted at compile (records, and the data scratch when
    /// the size is declared).
    #[must_use]
    pub const fn simulation_quote(&self) -> Option<SimulationQuote> {
        self.program.simulation()
    }

    /// Solve the plan, and simulate it when the request asks for it.
    ///
    /// The simulation is quoted at the planned size (one record per replicate
    /// and one data scratch per worker) and leased from `ctx`'s admission pool
    /// before anything is allocated; replicates run under `ctx`'s workers and
    /// reduce in replicate order, so the result does not depend on the thread
    /// count.
    ///
    /// # Errors
    ///
    /// A planning refusal (a target the design does not reach, a size below
    /// the design's minimum), [`CausalError::Resource`] when the pool cannot
    /// hold the simulation, or cancellation.
    pub fn execute(&self, ctx: &ExecutionContext) -> Result<PlannedExperiment, CausalError> {
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: "planning" });
        }
        let scoped = ctx.enter_lowered_dispatch(self.identity());
        let request = self.program.request();
        let mut result = plan_analytic(request).map_err(|error| refusal(&error))?;
        if let Some(spec) = request.simulation {
            let workers = u64::from(scoped.parallelism.max_threads.get());
            let quote = SimulationQuote::quote(
                u64::from(spec.replicates),
                scratch_bytes(request, &result),
                workers,
            )
            .ok_or_else(|| CausalError::Resource {
                message: format!(
                    "the planning simulation store of {} replicates overflows u64",
                    spec.replicates
                ),
            })?;
            let lease = scoped.acquire_admission(0, quote.total_bytes()).map_err(|error| {
                CausalError::Resource {
                    message: format!(
                        "the planning simulation needs {} bytes ({} replicate records of {} \
                         bytes and {} worker scratches of {} bytes) and the admission pool \
                         refused them: {error}; next attempt: lower the replicates or the \
                         threads, or raise the pool",
                        quote.total_bytes(),
                        quote.replicates(),
                        quote.record_bytes(),
                        quote.workers(),
                        quote.scratch_bytes(),
                    ),
                }
            })?;
            let simulated = simulate(request, &result, spec, &scoped);
            drop(lease);
            result.simulation = Some(simulated?);
        }
        Ok(PlannedExperiment { program_identity: self.identity(), result })
    }
}

/// Map a planning refusal to the reason-coded compile refusal it is.
fn refusal(refusal: &PlanningRefusal) -> CausalError {
    CausalError::Compile { message: refusal.to_string() }
}

/// Validate `request` and quote the part of its simulation store compile can
/// size, refusing a store the admission pool cannot hold.
fn planning_program(
    request: &PlanningRequest,
    ctx: &ExecutionContext,
) -> Result<PlanningProgram, CausalError> {
    refuse_reentry(ctx)?;
    if ctx.cancellation.is_cancelled() {
        return Err(CausalError::Cancelled { stage: "compile" });
    }
    request.validate().map_err(|error| refusal(&error))?;
    let simulation = match request.simulation {
        None => None,
        Some(spec) => {
            let scratch = match request.goal {
                PlanningGoal::Evaluate { units }
                | PlanningGoal::MinimumDetectableEffect { units } => individuals(request, units)
                    .saturating_mul(48 + 8 * u64::from(request.assumptions.covariates)),
                PlanningGoal::SampleSize { .. } => 0,
            };
            let workers = u64::from(ctx.parallelism.max_threads.get());
            let quote = SimulationQuote::quote(u64::from(spec.replicates), scratch, workers)
                .ok_or_else(|| CausalError::Resource {
                    message: format!(
                        "the planning simulation store of {} replicates overflows u64",
                        spec.replicates
                    ),
                })?;
            let available = ctx.admission().available().bytes();
            if quote.total_bytes() > available {
                return Err(CausalError::Resource {
                    message: format!(
                        "the planning simulation store of {} replicates needs {} bytes and the \
                         admission pool holds {available}; next attempt: lower the replicates \
                         or raise the pool",
                        quote.replicates(),
                        quote.total_bytes()
                    ),
                });
            }
            Some(quote)
        }
    };
    Ok(PlanningProgram {
        request: Arc::new(request.clone()),
        request_identity: request.identity(),
        simulation,
    })
}

/// Individuals a declared size enrols, as a saturating byte-quote bound.
fn individuals(request: &PlanningRequest, units: u64) -> u64 {
    match request.effective_design() {
        PlannedDesign::Cluster { mean_cluster_size, cluster_size_cv, .. } => {
            saturating_count(units as f64 * mean_cluster_size * (1.0 + 4.0 * cluster_size_cv))
        }
        PlannedDesign::Switchback { sequences, periods_per_block, .. } => {
            units.saturating_mul(sequences).saturating_mul(periods_per_block)
        }
        _ => units,
    }
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "a finite, nonnegative byte bound saturates at u64::MAX"
)]
fn saturating_count(x: f64) -> u64 {
    x.ceil().max(0.0) as u64
}

/// The statistical program of a plan: the analysis the design is read with.
fn statistical_program(request: &PlanningRequest) -> Result<StatisticalProgram, CausalError> {
    let declaration = |error: antecedent_core::DeclarationError| CausalError::Compile {
        message: error.to_string(),
    };
    let analysis = format!("planning.{}", request.design.analysis(&request.assumptions));
    let (realization, posterior) = if request.criterion.is_bayesian() {
        let prior = Prior::new(format!("planning.{}.normal_prior", request.criterion.as_str()))
            .map_err(declaration)?;
        let likelihood = LikelihoodModel::new(analysis).map_err(declaration)?;
        (
            Realization::Bayesian(OrdinaryBayesianLikelihood::new(likelihood, prior.clone())),
            Some(PosteriorPlan { algorithm: PosteriorAlgorithm::Conjugate, draws: 0, prior }),
        )
    } else {
        (
            Realization::Frequentist(ScoreFrequentist::new(
                ScoreModel::new(analysis).map_err(declaration)?,
            )),
            None,
        )
    };
    Ok(StatisticalProgram {
        identifier: None,
        estimator: None,
        realization,
        models: Arc::from(vec![ModelSlot {
            role: ModelRole::Outcome,
            family: Arc::from(format!(
                "planning.{}.{}",
                request.assumptions.outcome.as_str(),
                request.design.as_str()
            )),
            inputs: Arc::from(Vec::new()),
            output: None,
        }]),
        folds: None,
        resampling: None,
        posterior,
        design: DesignShape::not_design_based(PLANNING_DESIGN_SHAPE),
        operators: Arc::from(Vec::new()),
    })
}

/// Lower a standalone plan.
fn lower_planning(program: PlanningProgram) -> Result<LoweredProgram, CausalError> {
    let request = program.request();
    let design = request.design.as_str();
    let data = match request.design {
        PlannedDesign::Switchback { .. } => DataClassification::Temporal,
        _ => DataClassification::Tabular,
    };
    let scientific = ScientificProblem {
        target: ProblemTarget::QueryFunctional {
            kind: FunctionalKind::AssignmentContrast,
            outcome: None,
            query: *program.request_identity().as_bytes(),
        },
        plan_id: Arc::from(format!("planning.{design}")),
        data,
        graph_class: "none",
        query_variables: Arc::from(Vec::new()),
        source_digest: None,
        support_coordinate: None,
        licensed: false,
    };
    let statistical = statistical_program(request)?;
    let uncertainty = if request.criterion.is_bayesian() {
        UncertaintyReading::Posterior
    } else {
        UncertaintyReading::Sampling
    };
    let evidence = EvidenceDecisionProgram {
        sources: Arc::from(vec![EvidenceSourceRef { role: EvidenceRole::Trial, digest: None }]),
        composition: Composition::Single,
        decision: None,
        planning: Some(program),
        chain: None,
    };
    Ok(PlanningLoweredParts {
        design,
        scientific,
        identification: IdentificationLayer::Deferred { reason: PLANNING_IDENTIFIES_BY_DESIGN },
        statistical,
        evidence,
        physical: PhysicalLayer {
            operators: Arc::from(Vec::new()),
            design: DesignShape::not_design_based(PLANNING_DESIGN_SHAPE),
        },
        claim: ClaimDemand { uncertainty, kind: None },
    }
    .finish())
}

/// Compile a planning request for a future trial that binds no data.
///
/// The returned plan's lowered program carries the request (every planning
/// assumption, the criterion, the goal and the simulation size and seed) in
/// layer four; any change to them changes the program identity. Nothing is
/// solved or simulated.
///
/// # Errors
///
/// - a reason-coded refusal of the request: `invalid_argument`,
///   `required_option_missing` (interference without an exposure model, a
///   level-dependent analysis without a control mean),
///   `adaptive_contract_incomplete` (an adaptive design without a
///   pre-specified rule), `effect_not_identified` (carryover without washout),
///   `option_not_applicable` or `route_not_supported`, each naming the next
///   attempt where one exists;
/// - [`CausalError::Resource`]: the simulation's replicate store overflows or
///   exceeds the admission pool;
/// - `public_builder_reentry` inside a lowered dispatch, or cancellation.
pub fn compile_planning(
    request: &PlanningRequest,
    ctx: &ExecutionContext,
) -> Result<PlanningPlan, CausalError> {
    let program = planning_program(request, ctx)?;
    let lowered = lower_planning(program.clone())?;
    Ok(PlanningPlan { lowered: Arc::new(lowered), program })
}

/// Whether a study's declared randomization is the planned design.
fn randomization_matches(declared: &RandomizationDesign, planned: &PlannedDesign) -> bool {
    matches!(
        (declared, planned),
        (RandomizationDesign::Bernoulli, PlannedDesign::Bernoulli)
            | (
                RandomizationDesign::Complete { .. },
                PlannedDesign::Complete | PlannedDesign::GroupSequential { .. }
            )
            | (RandomizationDesign::Cluster { .. }, PlannedDesign::Cluster { .. })
            | (RandomizationDesign::Stratified { .. }, PlannedDesign::Stratified { .. })
            | (RandomizationDesign::Factorial2x2 { .. }, PlannedDesign::Factorial { .. })
            | (RandomizationDesign::MultiArm { .. }, PlannedDesign::MultiArm { .. })
            | (RandomizationDesign::Switchback { .. }, PlannedDesign::Switchback { .. })
    )
}

impl Study {
    /// Compile the study for planning the next trial of its randomized
    /// analysis.
    ///
    /// The plan's lowered program is the one [`Self::compile`] returns, with
    /// the planning request (every assumption, the criterion, the goal and the
    /// simulation size and seed) in layer four. The study must be a randomized
    /// experiment whose declared randomization is the planned design. Nothing
    /// is fitted, solved or simulated.
    ///
    /// # Errors
    ///
    /// The refusals of [`compile_planning`]; `route_not_supported` naming
    /// [`STANDALONE_PLANNING`] when the study is not a randomized experiment
    /// or its randomization is not the planned design; any refusal of
    /// [`Self::compile`].
    pub fn compile_for_planning(
        &self,
        request: &PlanningRequest,
        ctx: &ExecutionContext,
    ) -> Result<PlanningPlan, CausalError> {
        let program = planning_program(request, ctx)?;
        let matches = match self.route_input()? {
            RouteInput::RandomizedEffect { query, .. } => {
                randomization_matches(&query.design, &request.effective_design())
            }
            _ => false,
        };
        if !matches {
            return Err(CausalError::Compile {
                message: PlanningRefusal {
                    code: antecedent_core::reason_code!("route_not_supported"),
                    detail: "planning.study",
                    message: format!(
                        "the study is not a randomized experiment whose randomization is the \
                         planned {} design",
                        request.design.as_str()
                    ),
                    next_attempt: Some(STANDALONE_PLANNING),
                }
                .to_string(),
            });
        }
        let plan = self.compile_for_goal(&CompileGoal::Planning(program.clone()), ctx)?;
        let lowered = plan.lowered.clone().ok_or_else(|| {
            crate::compile_reason!("route_not_lowered", "the study compiled without a program")
        })?;
        Ok(PlanningPlan { lowered, program })
    }
}
