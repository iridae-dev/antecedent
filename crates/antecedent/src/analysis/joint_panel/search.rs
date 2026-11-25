//! Native search handoff retaining the compiled source and its scientific permission.
//!
//! Every compiled search lowers a `JointPanelProgramSearch` program that is a child of the
//! fitted response's program. A public execution enters that program's dispatch, passes the
//! release support gate and runs the unchanged search engine. The responses a continuous or
//! certified search evaluates for its proposals are retargeted inside that dispatch as
//! children of the search program.
// SPDX-License-Identifier: MIT OR Apache-2.0
use super::compiled::{self, Entry};
use super::{
    JointPanelError, JointPanelFunctional, JointPanelResponseFunctional, JointPanelResult,
    JointPanelScientificScope, JointPanelWorld, engine, joint_panel_target, source,
};
use crate::analysis::lowering::joint_panel::search::{self as lowering, SearchOperation};
use crate::planner::lowered::{CarrierDemand, DecisionProgram, LoweredProgram};
use crate::planner::{PlanStage, PlanView};
use antecedent_core::{CanonicalWriter, ExecutionContext, ScientificWorldId, SemanticDigest};
use antecedent_design::decision_contract::ProgramSearchContract;
use antecedent_design::program_search::{FiniteProgramDomain, PreparedFiniteProgramSearch};
use antecedent_design::scientific_source::{
    ScientificDecisionSource, ScientificSourcePayload, ScopedProgramSearchResult,
};
use std::sync::Arc;

mod certified;
pub use certified::{
    JointPanelCertifiedSearch, JointPanelCertifiedSearchResult, JointPanelProgramSearchResult,
    replay_panel_certificate,
};

/// Numerical representation requested by the existing decision engine.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JointPanelSearchUncertainty {
    /// Point means; only admissible affine expected-utility queries are supported.
    PointMeans,
    /// Complete aligned fitted-response law, with its original uncertainty meaning.
    AlignedLaw,
}
/// Exact response functional, numerical representation, scientific scope and finite search space.
#[derive(Clone, Debug)]
pub struct JointPanelSearchRequest {
    /// Explicit model-conditional or identified causal consumption.
    pub scope: JointPanelScientificScope,
    /// Same prespecified time functional for every actual program.
    pub functional: JointPanelResponseFunctional,
    /// Mean-only versus complete aligned-law requirements.
    pub uncertainty: JointPanelSearchUncertainty,
    /// Existing decision objective and operational/risk demands.
    pub contract: ProgramSearchContract,
    /// Explicit bounds, costs and restrictions on the actual compiled schedules.
    pub domain: FiniteProgramDomain,
    /// Declared common utility units.
    pub utility_units: String,
    /// Caller bound on exhaustive finite enumeration.
    pub max_candidates: usize,
}
/// Frozen adapter over the existing finite-search owner and the checked compiler source.
#[derive(Clone, Debug)]
pub struct JointPanelSearch {
    plan: PreparedFiniteProgramSearch,
    source: Arc<ScientificDecisionSource>,
    scope: JointPanelScientificScope,
    retention: Arc<antecedent_core::execution::AdmissionLease>,
    program: SearchProgram,
}
impl JointPanelResult {
    /// Compile search from actual channel schedules, response law and scientific authority.
    /// No response fit is repeated. Later execution cannot silently discard the source scope.
    ///
    /// The search lowers a program-search program that is a child of this result's program.
    ///
    /// # Errors
    /// Missing scientific permission, incomplete uncertainty, mismatched controls or resources;
    /// `public_builder_reentry` inside a lowered dispatch.
    pub fn compile_search(
        &self,
        request: &JointPanelSearchRequest,
        ctx: &ExecutionContext,
    ) -> Result<JointPanelSearch, JointPanelError> {
        Entry::Public.admit(ctx)?;
        source::check_cancelled(ctx)?;
        let n = match request.uncertainty {
            JointPanelSearchUncertainty::PointMeans => 1,
            JointPanelSearchUncertainty::AlignedLaw => self.draws.len(),
        };
        let bytes = n
            .checked_mul(self.request().worlds.len())
            .and_then(|n| n.checked_mul(8))
            .and_then(|n| u64::try_from(n).ok())
            .ok_or_else(|| JointPanelError::compile("search response size overflow"))?;
        source::check_memory(ctx, bytes)?;
        if request.uncertainty == JointPanelSearchUncertainty::PointMeans {
            self.require_point_fit_diagnostics()?;
        }
        let scoped = match request.uncertainty {
            JointPanelSearchUncertainty::PointMeans => {
                self.project_mean_source(&request.functional)?
            }
            JointPanelSearchUncertainty::AlignedLaw => {
                self.project_law_source(&request.functional)?
            }
        };
        scoped.authorize(request.scope).map_err(map_scientific_error)?;
        let bytes = retained_source_quote(&scoped)?;
        source::check_memory(ctx, bytes)?;
        let retention = Arc::new(
            ctx.acquire_admission(0, bytes)
                .map_err(|error| engine::PanelError::Resource(error.to_string()))?,
        );
        let plan = self.compile_numerical_search(request, ctx)?;
        let program = SearchSpec::finite(self, request, bytes).lower(&request.contract)?;
        Ok(JointPanelSearch {
            plan,
            source: Arc::new(scoped),
            scope: request.scope,
            retention,
            program,
        })
    }
}
impl JointPanelSearch {
    /// Execute the original finite-search engine under the retained scientific scope, inside
    /// the dispatch of this search's program.
    ///
    /// # Errors
    /// Cancellation, resources, source demand or scientific permission is refused before a certificate.
    /// `public_builder_reentry` inside another lowered dispatch, and `not_supported` from a
    /// release build that does not support the route.
    pub fn execute(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<ScopedProgramSearchResult, JointPanelError> {
        let scoped = self.program.enter(ctx)?;
        self.retention.observe_current();
        self.source.search(&self.plan, self.scope, &scoped).map_err(map_scientific_error)
    }
    /// Change downstream utility/risk while retaining candidates, fitted response and authority.
    /// The search lowers again for the new objective.
    ///
    /// # Errors
    /// A changed input space or invalid replacement objective is refused by the existing owner.
    pub fn with_objective(&self, contract: ProgramSearchContract) -> Result<Self, JointPanelError> {
        // The existing owner's refusal of the replacement objective comes first.
        let program = self.program.with_objective(&contract);
        let plan = self.plan.with_objective(contract).map_err(JointPanelError::compile)?;
        Ok(Self {
            plan,
            source: Arc::clone(&self.source),
            scope: self.scope,
            retention: Arc::clone(&self.retention),
            program: program?,
        })
    }
    /// The lowered program this search executes.
    #[must_use]
    pub const fn lowered(&self) -> &Arc<LoweredProgram> {
        &self.program.lowered
    }
    /// One view over every layer of the lowered program this search executes. Reading it
    /// evaluates and searches nothing.
    #[must_use]
    pub fn plan_view(&self) -> PlanView {
        self.program.plan_view()
    }
    /// Identity of the lowered program this search executes.
    #[must_use]
    pub fn program_identity(&self) -> SemanticDigest {
        self.program.identity()
    }
    /// Exact scientific source, including typed target and retained identification obligations.
    #[must_use]
    pub fn scientific_source(&self) -> &ScientificDecisionSource {
        &self.source
    }
    /// Explicit numerical-plan extraction; this component alone grants no causal permission.
    #[must_use]
    pub fn numerical_plan(&self) -> &PreparedFiniteProgramSearch {
        &self.plan
    }
}
fn retained_source_quote(source: &ScientificDecisionSource) -> Result<u64, JointPanelError> {
    let bytes = match source.as_numerical_source() {
        ScientificSourcePayload::Means(means) => {
            let mut n = means
                .means
                .len()
                .checked_mul(8)
                .ok_or_else(|| JointPanelError::compile("mean source quote overflow"))?;
            for quantity in &means.coordinates {
                n = n
                    .checked_add(
                        std::mem::size_of_val(quantity)
                            + quantity.variable_id.len()
                            + quantity.variable_name.len()
                            + quantity.units.len()
                            + quantity.population_id.len()
                            + quantity.regime_id.len()
                            + quantity.functional_id.len()
                            + quantity.transform_id.len(),
                    )
                    .ok_or_else(|| JointPanelError::compile("coordinate quote overflow"))?;
            }
            n
        }
        ScientificSourcePayload::Law(law) => antecedent_io::to_cbor(law.metadata())
            .map_err(JointPanelError::compile)?
            .len()
            .checked_mul(2)
            .and_then(|n| n.checked_add(law.draws().len().checked_mul(8)?))
            .ok_or_else(|| JointPanelError::compile("law source quote overflow"))?,
    };
    u64::try_from(bytes).map_err(JointPanelError::compile)
}

/// Domain tag of the retained source a search reads from its parent fit.
const SEARCH_SOURCE_TAG: &str = "antecedent.joint_panel.program_search.source.v1";
/// Domain tag of a finite search's declaration other than its objective contract.
const FINITE_DECLARATION_TAG: &str = "antecedent.joint_panel.program_search.finite.v1";
/// Domain tag of a continuous or certified search's declaration other than its objective.
const CONTINUOUS_DECLARATION_TAG: &str = "antecedent.joint_panel.program_search.continuous.v1";
/// Domain tag of the decision contract a search program records.
const SEARCH_CONTRACT_TAG: &str = "antecedent.joint_panel.program_search.contract.v1";

/// What a search program lowers from, other than its objective contract.
#[derive(Clone, Debug)]
struct SearchSpec {
    operation: SearchOperation,
    /// Program of the fitted response the search reads.
    parent: SemanticDigest,
    /// The retained source of the parent fit: its dependency, target and representation.
    source: SemanticDigest,
    /// Bytes of the retained source the search holds.
    source_bytes: u64,
    /// `Mean` for point means, `DecisionDraws` for the aligned law.
    carrier: CarrierDemand,
    /// Candidate programs: the declared worlds of a finite search, the evaluation bound of a
    /// continuous search, none enumerated for a certified search over its box.
    actions: usize,
    /// The budget, when declared, and the operational restrictions.
    domain_constraints: usize,
    utility_units: Arc<str>,
    /// Digest of the declaration other than the objective contract.
    declaration: [u8; 32],
}

/// A lowered search program and the declaration it lowers again from when the objective
/// changes.
#[derive(Clone, Debug)]
struct SearchProgram {
    spec: SearchSpec,
    lowered: Arc<LoweredProgram>,
}

impl SearchSpec {
    /// The declaration of a finite search over `result`'s declared worlds, holding a retained
    /// source of `source_bytes`.
    fn finite(
        result: &JointPanelResult,
        request: &JointPanelSearchRequest,
        source_bytes: u64,
    ) -> Self {
        let mut writer = CanonicalWriter::new(FINITE_DECLARATION_TAG);
        writer.text(scope_name(request.scope)).word(request.max_candidates as u64);
        write_search(
            &mut writer,
            &request.functional,
            request.uncertainty,
            &request.domain,
            &request.utility_units,
        );
        Self {
            operation: SearchOperation::Finite,
            parent: result.program_identity(),
            source: search_source(result, request.uncertainty),
            source_bytes,
            carrier: carrier(request.uncertainty),
            actions: result.request().worlds.len(),
            domain_constraints: domain_constraints(&request.domain),
            utility_units: Arc::from(request.utility_units.as_str()),
            declaration: writer.finish(),
        }
    }

    /// The declaration of a continuous (`Continuous`) or certified (`Certified`) search over
    /// programs proposed from `result`'s frozen fit.
    fn continuous(
        result: &JointPanelResult,
        request: &JointPanelContinuousSearchRequest,
        operation: SearchOperation,
    ) -> Self {
        let mut writer = CanonicalWriter::new(CONTINUOUS_DECLARATION_TAG);
        writer
            .text(operation.as_str())
            .text(scope_name(request.scope))
            .word(u64::from(request.base_world.raw()))
            .reals(&request.initial);
        write_search(
            &mut writer,
            &request.functional,
            request.uncertainty,
            &request.domain,
            &request.utility_units,
        );
        // Only the bounded poll reads the evaluation policy; a certificate bounds the box.
        let actions = if operation == SearchOperation::Continuous {
            let options = &request.options;
            writer
                .word(options.max_evaluations as u64)
                .word(options.max_iterations as u64)
                .real(options.initial_step)
                .real(options.step_tolerance)
                .real(options.objective_tolerance);
            options.max_evaluations
        } else {
            0
        };
        // Each proposal holds its response representation over one world.
        let values = match request.uncertainty {
            JointPanelSearchUncertainty::PointMeans => 1,
            JointPanelSearchUncertainty::AlignedLaw => result.draws.len(),
        };
        Self {
            operation,
            parent: result.program_identity(),
            source: search_source(result, request.uncertainty),
            source_bytes: (values as u64).saturating_mul(8),
            carrier: carrier(request.uncertainty),
            actions,
            domain_constraints: domain_constraints(&request.domain),
            utility_units: Arc::from(request.utility_units.as_str()),
            declaration: writer.finish(),
        }
    }

    /// Lower the search program for `contract`: the decision carries the carrier, the
    /// candidate programs as actions, the budget, restrictions and risk constraints as
    /// constraints, and the identity of the complete declaration and contract.
    fn lower(self, contract: &ProgramSearchContract) -> Result<SearchProgram, JointPanelError> {
        let identity = contract.identity().map_err(JointPanelError::compile)?;
        let mut writer = CanonicalWriter::new(SEARCH_CONTRACT_TAG);
        writer.bytes(&self.declaration).bytes(&identity);
        let decision = DecisionProgram {
            carrier: self.carrier,
            utility: DecisionProgram::utility_form(None),
            actions: self.actions,
            constraints: self.domain_constraints + contract.constraints.len(),
            chance_threshold: contract
                .constraints
                .iter()
                .map(|constraint| constraint.min_probability)
                .fold(0.0_f64, f64::max),
            utility_unit: Some(Arc::clone(&self.utility_units)),
            contract_identity: Some(Arc::from(writer.finish_hex())),
            information_value: None,
            scenario_source: None,
        };
        let lowered = lowering::lower(
            self.operation,
            Some(self.parent),
            self.source,
            self.source_bytes,
            decision,
        )?;
        Ok(SearchProgram { spec: self, lowered: Arc::new(lowered) })
    }
}

impl SearchProgram {
    /// The same search lowered for a replacement objective contract.
    fn with_objective(&self, contract: &ProgramSearchContract) -> Result<Self, JointPanelError> {
        self.spec.clone().lower(contract)
    }

    /// Identity of the lowered program.
    fn identity(&self) -> SemanticDigest {
        self.lowered.identity()
    }

    /// One view over every layer of the lowered program, at the prepared stage.
    fn plan_view(&self) -> PlanView {
        PlanView::of_program(Arc::clone(&self.lowered), PlanStage::Prepared)
    }

    /// Enter this program's dispatch from a public execution entry point and pass its release
    /// support gate. The engines run with the returned context.
    fn enter(&self, ctx: &ExecutionContext) -> Result<ExecutionContext, JointPanelError> {
        let scoped = compiled::enter(ctx, &self.lowered)?;
        compiled::require_release_support(&scoped, self.spec.operation.route())?;
        Ok(scoped)
    }
}

/// Identity of the retained source a search reads from `result`: the fit dependency, the
/// parent's scientific target and the representation the search reads.
fn search_source(
    result: &JointPanelResult,
    uncertainty: JointPanelSearchUncertainty,
) -> SemanticDigest {
    let mut writer = CanonicalWriter::new(SEARCH_SOURCE_TAG);
    writer
        .bytes(result.dependency_identity().as_bytes())
        .bytes(result.request().target.identity().as_bytes())
        .text(uncertainty_name(uncertainty));
    SemanticDigest::from_bytes(writer.finish())
}

/// The evidence the search decision reads: point means or the aligned law.
const fn carrier(uncertainty: JointPanelSearchUncertainty) -> CarrierDemand {
    match uncertainty {
        JointPanelSearchUncertainty::PointMeans => CarrierDemand::Mean,
        JointPanelSearchUncertainty::AlignedLaw => CarrierDemand::DecisionDraws,
    }
}

/// The budget, when declared, and every operational restriction.
fn domain_constraints(domain: &FiniteProgramDomain) -> usize {
    usize::from(domain.budget.is_some()) + domain.restrictions.len()
}

/// Stable name of the response representation a search reads.
const fn uncertainty_name(uncertainty: JointPanelSearchUncertainty) -> &'static str {
    match uncertainty {
        JointPanelSearchUncertainty::PointMeans => "point_means",
        JointPanelSearchUncertainty::AlignedLaw => "aligned_law",
    }
}

/// Stable name of the scientific scope a search consumes its source under.
const fn scope_name(scope: JointPanelScientificScope) -> &'static str {
    match scope {
        JointPanelScientificScope::ModelConditional => "model_conditional",
        JointPanelScientificScope::IdentifiedCausal => "identified_causal",
    }
}

/// Write the response functional, representation, utility units and complete program domain
/// a search declares.
fn write_search(
    writer: &mut CanonicalWriter,
    functional: &JointPanelResponseFunctional,
    uncertainty: JointPanelSearchUncertainty,
    domain: &FiniteProgramDomain,
    utility_units: &str,
) {
    match functional {
        JointPanelResponseFunctional::TerminalMean => {
            writer.text("terminal_mean");
        }
        JointPanelResponseFunctional::WindowAverage { from, until } => {
            writer.text("window_average").word(*from as u64).word(*until as u64);
        }
        JointPanelResponseFunctional::WindowChange { before, after } => {
            writer.text("window_change").word(before.0 as u64).word(before.1 as u64);
            writer.word(after.0 as u64).word(after.1 as u64);
        }
        JointPanelResponseFunctional::WeightedSum { coefficients } => {
            writer.text("weighted_sum").reals(coefficients);
        }
    }
    writer
        .text(uncertainty_name(uncertainty))
        .text(utility_units)
        .word(domain.controls.len() as u64);
    for control in &domain.controls {
        writer
            .word(u64::from(control.variable.raw()))
            .bytes(&control.offset.to_le_bytes())
            .real(control.lower)
            .real(control.upper)
            .real(control.unit_cost);
    }
    writer.opt_real(domain.budget).word(domain.restrictions.len() as u64);
    for restriction in &domain.restrictions {
        writer.text(&restriction.id).real(restriction.bound).reals(&restriction.coefficients);
    }
}

/// Bounded numerical next attempt after explicit finite-program evaluation.
#[derive(Clone, Debug)]
pub struct JointPanelContinuousSearchRequest {
    /// Exact retained world whose future context/history is held fixed while controls vary.
    pub base_world: ScientificWorldId,
    /// Explicit conditional use; new programs require their own evidence before causal promotion.
    pub scope: JointPanelScientificScope,
    /// Same time functional throughout the numerical procedure.
    pub functional: JointPanelResponseFunctional,
    /// Point means or complete aligned law.
    pub uncertainty: JointPanelSearchUncertainty,
    /// Existing separable objective and hard risk constraints.
    pub contract: ProgramSearchContract,
    /// Operational bounds, scheduled costs, budget and restrictions.
    pub domain: FiniteProgramDomain,
    /// Common utility units.
    pub utility_units: String,
    /// Explicit bounded poll policy.
    pub options: antecedent_design::program_search::continuous::ContinuousSearchOptions,
    /// One initial level per exact domain control; no absent offsets are filled.
    pub initial: Vec<f64>,
}
/// Continuous best-found search inseparable from the common retained panel fit.
#[derive(Clone, Debug)]
pub struct JointPanelContinuousSearch {
    plan: antecedent_design::program_search::continuous::PreparedContinuousProgramSearch,
    evaluator: FrozenPanelEvaluator,
    program: SearchProgram,
}
#[derive(Clone, Debug)]
struct FrozenPanelEvaluator {
    result: Arc<JointPanelResult>,
    base: JointPanelWorld,
    domain: FiniteProgramDomain,
    functional: JointPanelResponseFunctional,
    uncertainty: JointPanelSearchUncertainty,
    /// Identity of the search program inside whose dispatch proposals are evaluated; each
    /// proposal's retargeted response is a child of that program.
    program: SemanticDigest,
}
/// Best-found numerical search and its independently consumable checked response source.
#[derive(Clone, Debug)]
pub struct JointPanelContinuousSearchResult {
    /// Actual proposals, feasibility/failures, stop reason and absence of a global certificate.
    pub search: antecedent_design::program_search::continuous::ContinuousProgramSearchResult,
    /// Exact response/proof for the best feasible program, when one was found.
    pub best_source: Option<ScientificDecisionSource>,
    program: SemanticDigest,
}
impl JointPanelContinuousSearchResult {
    /// Identity of the lowered search program that produced this result.
    #[must_use]
    pub const fn program_identity(&self) -> SemanticDigest {
        self.program
    }
}
impl JointPanelResult {
    /// Compile continuous search over actual programs evaluated with this frozen fit realization.
    ///
    /// The search lowers a program-search program that is a child of this result's program.
    ///
    /// # Errors
    /// Invalid controls, unsupported causal extrapolation, source representation or search policy;
    /// `public_builder_reentry` inside a lowered dispatch.
    pub fn compile_continuous_search(
        &self,
        request: JointPanelContinuousSearchRequest,
        ctx: &ExecutionContext,
    ) -> Result<JointPanelContinuousSearch, JointPanelError> {
        use antecedent_design::program_search::{
            ProgramSearchScope, continuous::PreparedContinuousProgramSearch,
        };
        Entry::Public.admit(ctx)?;
        if request.scope != JointPanelScientificScope::ModelConditional {
            return Err(JointPanelError::Unsupported("continuous proposals need their own causal design proof; this model-based numerical next attempt grants only explicitly conditional use".into()));
        }
        let base = self
            .request()
            .worlds
            .iter()
            .find(|world| world.world == request.base_world)
            .cloned()
            .ok_or_else(|| JointPanelError::compile("continuous base world is not retained"))?;
        let finite = JointPanelSearchRequest {
            scope: request.scope,
            functional: request.functional.clone(),
            uncertainty: request.uncertainty,
            contract: request.contract.clone(),
            domain: request.domain.clone(),
            utility_units: request.utility_units.clone(),
            max_candidates: self.request().worlds.len(),
        };
        if request.uncertainty == JointPanelSearchUncertainty::PointMeans {
            self.require_point_fit_diagnostics()?;
        }
        // The same production control/quantity admission used by the finite next-attempt predecessor.
        self.compile_numerical_search(&finite, ctx)?;
        let means = self.numerical_project_mean_source(&request.functional)?;
        let coordinate = means
            .coordinates
            .first()
            .ok_or_else(|| JointPanelError::compile("no retained response coordinate"))?;
        let spec = SearchSpec::continuous(self, &request, SearchOperation::Continuous);
        let program = spec.lower(&request.contract)?;
        let plan = PreparedContinuousProgramSearch::compile(
            request.contract,
            request.domain.clone(),
            ProgramSearchScope {
                population: coordinate.population_id.clone(),
                horizon: coordinate.horizon,
                utility_units: request.utility_units,
                max_candidates: 1,
            },
            request.scope,
            request.options,
            request.initial,
        )
        .map_err(JointPanelError::compile)?;
        Ok(JointPanelContinuousSearch {
            plan,
            evaluator: FrozenPanelEvaluator {
                result: Arc::new(self.clone()),
                base,
                domain: request.domain,
                functional: request.functional,
                uncertainty: request.uncertainty,
                program: program.identity(),
            },
            program,
        })
    }
}
impl JointPanelContinuousSearch {
    /// Execute bounded polling through the retained compiler and existing decision engine,
    /// inside the dispatch of this search's program. Each proposal's response is retargeted
    /// from the frozen fit as a child of this program.
    ///
    /// # Errors
    /// Numerical source, resource or cancellation refusal; no global certificate is fabricated.
    /// `public_builder_reentry` inside another lowered dispatch, and `not_supported` from a
    /// release build that does not support the route.
    pub fn execute(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<JointPanelContinuousSearchResult, JointPanelError> {
        use antecedent_design::program_search::continuous::CompiledProgramEvaluator;
        let scoped = self.program.enter(ctx)?;
        let search = self.plan.execute(&self.evaluator, &scoped).map_err(map_search_error)?;
        let best_source = search
            .best
            .as_ref()
            .map(|best| {
                self.evaluator
                    .evaluate_program(&best.controls, &scoped)
                    .map(|compiled| compiled.source)
            })
            .transpose()
            .map_err(map_search_error)?;
        Ok(JointPanelContinuousSearchResult {
            search,
            best_source,
            program: self.program.identity(),
        })
    }
    /// Change only downstream objective/risk, preserving the exact fitted response realization.
    /// The search lowers again for the new objective.
    ///
    /// # Errors
    /// Changed response space, comparative objectives or invalid contracts are refused.
    pub fn with_objective(&self, contract: ProgramSearchContract) -> Result<Self, JointPanelError> {
        // The existing owner's refusal of the replacement objective comes first.
        let program = self.program.with_objective(&contract);
        let plan = self.plan.with_objective(contract).map_err(JointPanelError::compile)?;
        let program = program?;
        let mut evaluator = self.evaluator.clone();
        evaluator.program = program.identity();
        Ok(Self { plan, evaluator, program })
    }
    /// The lowered program this search executes.
    #[must_use]
    pub const fn lowered(&self) -> &Arc<LoweredProgram> {
        &self.program.lowered
    }
    /// One view over every layer of the lowered program this search executes. Reading it
    /// evaluates and searches nothing.
    #[must_use]
    pub fn plan_view(&self) -> PlanView {
        self.program.plan_view()
    }
    /// Identity of the lowered program this search executes.
    #[must_use]
    pub fn program_identity(&self) -> SemanticDigest {
        self.program.identity()
    }
}
impl antecedent_design::program_search::continuous::CompiledProgramEvaluator
    for FrozenPanelEvaluator
{
    fn evaluate_program(
        &self,
        controls: &[f64],
        ctx: &ExecutionContext,
    ) -> Result<
        antecedent_design::program_search::continuous::CompiledContinuousCandidate,
        antecedent_design::program_search::ProgramSearchError,
    > {
        use antecedent_design::program_search::{
            ProgramCandidate, continuous::CompiledContinuousCandidate,
        };
        let mut world = self.base.clone();
        let mut writer =
            antecedent_core::CanonicalWriter::new("antecedent.panel.continuous_program.v1");
        if controls.len() != self.domain.controls.len() {
            return Err(antecedent_design::program_search::ProgramSearchError::Invalid(
                "proposal omits scheduled controls".into(),
            ));
        }
        for (&value, control) in controls.iter().zip(&self.domain.controls) {
            let id = control.variable.raw().to_string();
            let channel =
                world.program.channels.iter_mut().find(|channel| channel.id == id).ok_or_else(
                    || {
                        antecedent_design::program_search::ProgramSearchError::Invalid(
                            "unknown compiled channel".into(),
                        )
                    },
                )?;
            let slot = usize::try_from(control.offset)
                .ok()
                .and_then(|offset| channel.values.get_mut(offset))
                .ok_or_else(|| {
                    antecedent_design::program_search::ProgramSearchError::Invalid(
                        "unknown compiled offset".into(),
                    )
                })?;
            *slot = value;
            writer
                .word(u64::from(control.variable.raw()))
                .word(u64::try_from(control.offset).map_err(|_| {
                    antecedent_design::program_search::ProgramSearchError::Invalid(
                        "negative compiled offset".into(),
                    )
                })?)
                .word(value.to_bits());
        }
        world.program.id = SemanticDigest::from_bytes(writer.finish()).to_hex();
        world.reference = true;
        let id = world.program.id.clone();
        let request = self.result.request();
        let target = joint_panel_target(
            self.result.prepared().source_data().schema(),
            &request.columns,
            request.lag_order,
            std::slice::from_ref(&world),
            JointPanelFunctional::Worlds,
            request.target.inference(),
        )
        .map_err(map_joint_error)?;
        // The proposal's response is a child of the search program whose dispatch `ctx` is in.
        let result = self
            .result
            .retarget_frozen_entry(Entry::Child(self.program), vec![world], target, ctx)
            .map_err(map_joint_error)?;
        let source = match self.uncertainty {
            JointPanelSearchUncertainty::PointMeans => result.project_mean_source(&self.functional),
            JointPanelSearchUncertainty::AlignedLaw => result.project_law_source(&self.functional),
        }
        .map_err(map_joint_error)?;
        let inputs = match source.as_numerical_source() {
            ScientificSourcePayload::Means(means) => means.coordinates.clone(),
            ScientificSourcePayload::Law(law) => law
                .metadata()
                .identity
                .quantities
                .iter()
                .cloned()
                .map(antecedent_contract::ScientificQuantity::try_from)
                .collect::<Result<Vec<_>, _>>()
                .map_err(|error| {
                    antecedent_design::program_search::ProgramSearchError::Invalid(
                        error.to_string(),
                    )
                })?,
        };
        Ok(CompiledContinuousCandidate {
            candidate: ProgramCandidate { id, controls: controls.to_vec(), inputs },
            source,
        })
    }
}
fn map_joint_error(
    error: JointPanelError,
) -> antecedent_design::program_search::ProgramSearchError {
    use antecedent_design::program_search::ProgramSearchError;
    match error {
        JointPanelError::Engine(engine::PanelError::Resource(error)) => {
            ProgramSearchError::Resource(error)
        }
        JointPanelError::Engine(engine::PanelError::Cancelled) => ProgramSearchError::Cancelled,
        error => ProgramSearchError::Invalid(error.to_string()),
    }
}
fn map_search_error(
    error: antecedent_design::program_search::ProgramSearchError,
) -> JointPanelError {
    use antecedent_design::program_search::ProgramSearchError;
    match error {
        ProgramSearchError::Resource(error) => engine::PanelError::Resource(error).into(),
        ProgramSearchError::Budget { summary, .. } => engine::PanelError::Resource(summary).into(),
        ProgramSearchError::Cancelled => engine::PanelError::Cancelled.into(),
        error @ (ProgramSearchError::Invalid(_) | ProgramSearchError::Upstream { .. }) => {
            JointPanelError::compile(error)
        }
    }
}

pub(super) fn map_scientific_error(
    error: antecedent_design::scientific_source::ScientificSourceError,
) -> JointPanelError {
    use antecedent_design::scientific_source::ScientificSourceError;
    match error {
        ScientificSourceError::Search(error) => map_search_error(error),
        ScientificSourceError::Authorization(error) => JointPanelError::Unsupported(error),
        ScientificSourceError::Decision(error) => JointPanelError::compile(format!("{error:?}")),
    }
}

impl crate::analysis::program_handle::CompiledProgramHandle for JointPanelSearch {
    fn program(&self) -> &std::sync::Arc<crate::planner::lowered::LoweredProgram> {
        &self.program.lowered
    }
}

impl crate::analysis::program_handle::CompiledProgramHandle for JointPanelContinuousSearch {
    fn program(&self) -> &std::sync::Arc<crate::planner::lowered::LoweredProgram> {
        &self.program.lowered
    }
}

#[cfg(test)]
#[path = "search_site_tests.rs"]
mod site_tests;
