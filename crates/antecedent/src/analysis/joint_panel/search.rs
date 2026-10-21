//! Native search handoff retaining the compiled source and its scientific permission.
// SPDX-License-Identifier: MIT OR Apache-2.0
use super::{
    JointPanelError, JointPanelFunctional, JointPanelResponseFunctional, JointPanelResult,
    JointPanelScientificScope, JointPanelWorld, engine, joint_panel_target, source,
};
use antecedent_core::{ExecutionContext, ScientificWorldId, SemanticDigest};
use antecedent_design::decision_contract::ProgramSearchContract;
use antecedent_design::program_search::{FiniteProgramDomain, PreparedFiniteProgramSearch};
use antecedent_design::scientific_source::{
    ScientificDecisionSource, ScientificSourcePayload, ScopedProgramSearchResult,
};
use std::sync::Arc;

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
}
impl JointPanelResult {
    /// Compile search from actual channel schedules, response law and scientific authority.
    /// No response fit is repeated. Later execution cannot silently discard the source scope.
    ///
    /// # Errors
    /// Missing scientific permission, incomplete uncertainty, mismatched controls or resources.
    pub fn compile_search(
        &self,
        request: &JointPanelSearchRequest,
        ctx: &ExecutionContext,
    ) -> Result<JointPanelSearch, JointPanelError> {
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
        Ok(JointPanelSearch { plan, source: Arc::new(scoped), scope: request.scope, retention })
    }
}
impl JointPanelSearch {
    /// Execute the original finite-search engine under the retained scientific scope.
    ///
    /// # Errors
    /// Cancellation, resources, source demand or scientific permission is refused before a certificate.
    pub fn execute(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<ScopedProgramSearchResult, JointPanelError> {
        self.retention.observe_current();
        self.source.search(&self.plan, self.scope, ctx).map_err(map_scientific_error)
    }
    /// Change downstream utility/risk while retaining candidates, fitted response and authority.
    ///
    /// # Errors
    /// A changed input space or invalid replacement objective is refused by the existing owner.
    pub fn with_objective(&self, contract: ProgramSearchContract) -> Result<Self, JointPanelError> {
        Ok(Self {
            plan: self.plan.with_objective(contract).map_err(JointPanelError::compile)?,
            source: Arc::clone(&self.source),
            scope: self.scope,
            retention: Arc::clone(&self.retention),
        })
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
}
#[derive(Clone, Debug)]
struct FrozenPanelEvaluator {
    result: Arc<JointPanelResult>,
    base: JointPanelWorld,
    domain: FiniteProgramDomain,
    functional: JointPanelResponseFunctional,
    uncertainty: JointPanelSearchUncertainty,
}
/// Best-found numerical search and its independently consumable checked response source.
#[derive(Clone, Debug)]
pub struct JointPanelContinuousSearchResult {
    /// Actual proposals, feasibility/failures, stop reason and absence of a global certificate.
    pub search: antecedent_design::program_search::continuous::ContinuousProgramSearchResult,
    /// Exact response/proof for the best feasible program, when one was found.
    pub best_source: Option<ScientificDecisionSource>,
}
impl JointPanelResult {
    /// Compile continuous search over actual programs evaluated with this frozen fit realization.
    ///
    /// # Errors
    /// Invalid controls, unsupported causal extrapolation, source representation or search policy.
    pub fn compile_continuous_search(
        &self,
        request: JointPanelContinuousSearchRequest,
        ctx: &ExecutionContext,
    ) -> Result<JointPanelContinuousSearch, JointPanelError> {
        use antecedent_design::program_search::{
            ProgramSearchScope, continuous::PreparedContinuousProgramSearch,
        };
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
        // The same production control/quantity admission used by the finite next-attempt predecessor.
        self.compile_numerical_search(&finite, ctx)?;
        let means = self.numerical_project_mean_source(&request.functional)?;
        let coordinate = means
            .coordinates
            .first()
            .ok_or_else(|| JointPanelError::compile("no retained response coordinate"))?;
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
            },
        })
    }
}
impl JointPanelContinuousSearch {
    /// Execute bounded polling through the retained compiler and existing decision engine.
    ///
    /// # Errors
    /// Numerical source, resource or cancellation refusal; no global certificate is fabricated.
    pub fn execute(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<JointPanelContinuousSearchResult, JointPanelError> {
        use antecedent_design::program_search::continuous::CompiledProgramEvaluator;
        let search = self.plan.execute(&self.evaluator, ctx).map_err(map_search_error)?;
        let best_source = search
            .best
            .as_ref()
            .map(|best| {
                self.evaluator.evaluate_program(&best.controls, ctx).map(|compiled| compiled.source)
            })
            .transpose()
            .map_err(map_search_error)?;
        Ok(JointPanelContinuousSearchResult { search, best_source })
    }
    /// Change only downstream objective/risk, preserving the exact fitted response realization.
    ///
    /// # Errors
    /// Changed response space, comparative objectives or invalid contracts are refused.
    pub fn with_objective(&self, contract: ProgramSearchContract) -> Result<Self, JointPanelError> {
        Ok(Self {
            plan: self.plan.with_objective(contract).map_err(JointPanelError::compile)?,
            evaluator: self.evaluator.clone(),
        })
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
        let result =
            self.result.retarget_frozen(vec![world], target, ctx).map_err(map_joint_error)?;
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
        ProgramSearchError::Cancelled => engine::PanelError::Cancelled.into(),
        error @ ProgramSearchError::Invalid(_) => JointPanelError::compile(error),
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
