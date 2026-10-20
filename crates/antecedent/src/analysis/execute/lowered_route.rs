//! Retained linear/Gaussian intervention route.
//!
//! Lowers the existing static `linear.adjustment.ate` contrast through the six
//! compiler layers and executes the checked program that lowering retains.
//! [`super::super::Study::compile`] builds the physical plan this route keeps.
//! Admission, identification, and preparation stay here.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![cfg_attr(not(test), allow(dead_code))]

use std::cell::RefCell;
use std::sync::Arc;

use antecedent_contract::ExecutionPerformanceRecord;
use antecedent_core::execution::{AdmissionError, AdmissionLease};
use antecedent_core::{
    Assumption, AssumptionRecord, AssumptionSet, AssumptionStatus, AverageEffectQuery, CausalQuery,
    EncodedWidth, ExecutionContext, Operator, RouteBudget, RouteMemoryDeclaration, ScientificTarget,
};
use antecedent_data::{design_layout::DesignLayout, GlobalId, TableView, TabularData};
use antecedent_estimate::{
    CheckedLinearAdjustmentAte, EffectEstimate, EstimationWorkspace, LinearAdjustmentAte,
    LinearFitKind, PreparedEstimationProblem,
};
use antecedent_io::{
    AnalysisResultWire, CausalQueryWire, IdentificationResultWire, causal_query_to_wire,
    encode_analysis_result_artifact, identification_to_wire,
};
use antecedent_stats::CompiledDesign;
use antecedent_graph::Dag;
use antecedent_identify::{
    AssembledFunctionalObligation, CompiledIdentificationProduct, CompiledProductParts,
    IdentificationClaimClass, IdentificationOutcome, IdentificationResult, RequiredFactor,
    UnresolvedObligation, WorldObligation,
};

use crate::analysis::prepared::recompile::{LifecycleInput, LifecycleProduct, PreparedRecompile};
use crate::error::CausalError;
use crate::estimator_spec::EstimatorSpec;
use crate::planner::{
    PhysicalExecutionPlan, linear_gaussian_physical_operators, linear_gaussian_program_operators,
    linear_gaussian_scientific_problem,
};
use crate::strategy_table::{
    EstimatorId, IdentifierId, identify_static_query_with_rd, select_claim,
};
use crate::support::CellStatus;

use super::super::Study;

const CONTROL_WORLD: u32 = 0;
const ACTIVE_WORLD: u32 = 1;

/// Decision request compiled for this route.
///
/// The route asks for the contrast mean. It does not ask for a utility, a
/// constraint, or a value of information.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum LinearGaussianDecision {
    /// Mean of the active-minus-control contrast.
    ContrastMean,
}

/// Evidence program: support status, provenance/execution/causal edges, and the
/// decision request. Assumptions stay on the identification product.
#[derive(Clone, Debug)]
pub(crate) struct LinearGaussianEvidenceProgram {
    support_status: Option<CellStatus>,
    operators: Vec<Operator>,
    decision: LinearGaussianDecision,
}

impl LinearGaussianEvidenceProgram {
    /// Support status copied from the study that was lowered.
    #[must_use]
    pub(crate) const fn support_status(&self) -> Option<CellStatus> {
        self.support_status
    }

    /// Read, fit, and intervene edges.
    #[must_use]
    pub(crate) fn operators(&self) -> &[Operator] {
        &self.operators
    }

    /// The decision request. This route has one.
    #[must_use]
    pub(crate) const fn decision(&self) -> LinearGaussianDecision {
        self.decision
    }
}

/// Claim produced by executing the retained program.
#[derive(Clone, Debug)]
pub(crate) struct LinearGaussianClaim {
    /// Point and uncertainty from the retained fit.
    pub(crate) estimate: EffectEstimate,
    /// Support status the study carried when it was lowered.
    pub(crate) support_status: Option<CellStatus>,
    /// Execution facts. Byte estimates stay on the physical plan.
    pub(crate) receipt: ExecutionPerformanceRecord,
}

/// Static linear/Gaussian intervention route, retained through all six layers.
///
/// Execute fits this program. It does not build another study.
#[derive(Clone, Debug)]
#[must_use = "execute the retained linear/Gaussian program"]
pub(crate) struct LoweredLinearGaussianRoute {
    scientific: ScientificTarget,
    identification: CompiledIdentificationProduct,
    /// Identification the artifact consumer reloads. The compiled product does not encode it.
    identified: IdentificationResult,
    /// Source table. Execute reads it through column views.
    data: TabularData,
    program: CheckedLinearAdjustmentAte,
    fitter: LinearAdjustmentAte,
    /// The id-selected route fits the point, then bootstraps. A configured OLS
    /// fitter already bootstraps inside `fit_checked`.
    point_then_bootstrap: bool,
    evidence: LinearGaussianEvidenceProgram,
    physical: PhysicalExecutionPlan,
    physical_operators: Vec<antecedent_core::PhysicalOperator>,
    /// Products whose validity decides whether the next call fits.
    recompile: RefCell<PreparedRecompile>,
    /// Claim from the last causal fit. Absent until that fit succeeds.
    cached_claim: RefCell<Option<LinearGaussianClaim>>,
}

impl LoweredLinearGaussianRoute {
    /// Layer 1. The contrast of the two intervention worlds.
    #[must_use]
    pub(crate) const fn scientific(&self) -> &ScientificTarget {
        &self.scientific
    }

    /// Layer 2. Functional, derivation, assumptions, factors, and obligations.
    pub(crate) const fn identification(&self) -> &CompiledIdentificationProduct {
        &self.identification
    }

    /// Layer 3. Checked linear program. Execution uses this value.
    #[must_use]
    pub(crate) const fn program(&self) -> &CheckedLinearAdjustmentAte {
        &self.program
    }

    /// Layer 4.
    #[must_use]
    pub(crate) const fn evidence(&self) -> &LinearGaussianEvidenceProgram {
        &self.evidence
    }

    /// Layer 5 record. The same record [`Study::compile`] builds for this study.
    #[must_use]
    pub(crate) const fn physical(&self) -> &PhysicalExecutionPlan {
        &self.physical
    }

    /// Layer 5 operators.
    #[must_use]
    pub(crate) fn physical_operators(&self) -> &[antecedent_core::PhysicalOperator] {
        &self.physical_operators
    }

    /// Fit the retained program.
    ///
    /// A later call reuses the retained causal fit while that product is still
    /// valid. Utility and action changes leave it valid, so they perform no
    /// new causal fit. A fresh fit reads treatment and outcome through column
    /// views and the design layout, then runs the ordinary least-squares
    /// program this route already retained.
    ///
    /// Admission runs before that read. A cancelled context or a declaration
    /// that exceeds the route budget returns without a fit.
    ///
    /// # Errors
    ///
    /// The context is already cancelled, the route budget cannot hold the
    /// declared bytes, or the retained fit fails.
    pub(crate) fn execute(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<LinearGaussianClaim, CausalError> {
        if ctx.cancellation.is_cancelled() {
            return Err(cancelled());
        }
        let _lease = admit_gaussian(self, ctx)?;
        if self.recompile.borrow().reuses_causal_fit() {
            if let Some(claim) = self.cached_claim.borrow().clone() {
                self.recompile.borrow_mut().retain_decision();
                return Ok(claim);
            }
        }
        let problem = self.seam_problem()?;
        let mut workspace = EstimationWorkspace::default();
        let assumptions = self.program.required_assumptions().clone();
        let estimate = if self.point_then_bootstrap {
            let point = self.fitter.fit_point(&problem, &mut workspace, assumptions)?;
            if self.fitter.bootstrap_replicates == 0 || ctx.cancellation.is_cancelled() {
                point
            } else {
                self.fitter.attach_bootstrap(&problem, &mut workspace, ctx, point)?
            }
        } else {
            self.fitter.fit(&problem, &mut workspace, ctx, assumptions)?
        };
        let claim = LinearGaussianClaim {
            receipt: receipt(&estimate, self.fitter.bootstrap_replicates),
            support_status: self.evidence.support_status,
            estimate,
        };
        *self.cached_claim.borrow_mut() = Some(claim.clone());
        self.recompile.borrow_mut().retain_causal_fit();
        Ok(claim)
    }

    /// Bytes of the executed contrast as an analysis-result artifact.
    ///
    /// A later consumer reads those bytes with no handle to this route.
    ///
    /// # Errors
    ///
    /// Execute has not yet produced a claim, or the artifact encoder refuses.
    pub(crate) fn claim_artifact(&self) -> Result<Vec<u8>, CausalError> {
        let claim = self
            .cached_claim
            .borrow()
            .clone()
            .ok_or_else(|| compile_error("linear/Gaussian route has no executed claim"))?;
        let query = causal_query_to_wire(&self.identified.query)?;
        let identification = identification_to_wire(&self.identified)?;
        let standard_error = claim.estimate.se_analytic();
        let wire = analysis_wire(query, identification, claim.estimate.ate, standard_error);
        let names = TableView::schema(&self.data)
            .variables()
            .iter()
            .map(|variable| variable.name.to_string())
            .collect();
        let artifact = encode_analysis_result_artifact(&wire, names, "gaussian-contrast")?;
        let mut bytes = Vec::new();
        artifact.write_to(&mut bytes)?;
        Ok(bytes)
    }

    /// Apply one lifecycle change and drop a cached claim whose fit it invalidates.
    pub(crate) fn apply_input(&self, input: LifecycleInput) {
        let keep_fit = {
            let mut state = self.recompile.borrow_mut();
            state.apply(input);
            state.reuses_causal_fit()
        };
        if !keep_fit {
            self.cached_claim.borrow_mut().take();
        }
    }

    /// Whether `product` is still valid on this route.
    #[must_use = "an invalid product has to be rebuilt"]
    pub(crate) fn product_is_valid(&self, product: LifecycleProduct) -> bool {
        self.recompile.borrow().is_valid(product)
    }

    /// Causal fits this route has run.
    #[must_use = "compare this with the count from before the change"]
    pub(crate) fn causal_fit_count(&self) -> u64 {
        self.recompile.borrow().causal_fits()
    }
}

fn receipt(estimate: &EffectEstimate, requested: u32) -> ExecutionPerformanceRecord {
    let bootstrap = &estimate.uncertainty.bootstrap;
    ExecutionPerformanceRecord {
        bootstrap_replicates_requested: Some(requested),
        bootstrap_replicates_ok: bootstrap.replicates_ok,
        cancelled: bootstrap.cancelled,
        early_stopped: bootstrap.early_stopped,
        copy_count: 1,
        ..ExecutionPerformanceRecord::default()
    }
}

fn admit_gaussian(
    route: &LoweredLinearGaussianRoute,
    ctx: &ExecutionContext,
) -> Result<AdmissionLease, CausalError> {
    let rows = u64::try_from(route.data.row_count())
        .map_err(|_| compile_error("row count does not fit in u64"))?;
    let lowering = route.program.lowering();
    let columns = lowering.adjustment.len().saturating_add(2);
    let mut declaration = RouteMemoryDeclaration::new().rows(rows);
    for _ in 0..columns {
        declaration = declaration.encoded_column(EncodedWidth::float64());
    }
    let mask_bytes = u64::try_from(route.data.row_count().div_ceil(8))
        .map_err(|_| compile_error("mask bytes do not fit in u64"))?;
    let workspace = route.physical.record.workspace_bytes.ok_or_else(|| CausalError::Resource {
        message: "linear/Gaussian physical plan omitted workspace bytes".into(),
    })?;
    let width = u64::try_from(columns)
        .map_err(|_| compile_error("column count does not fit in u64"))?;
    let design_bytes = rows.checked_mul(width).and_then(|cells| cells.checked_mul(8)).ok_or_else(
        || CausalError::Resource { message: "linear/Gaussian design bytes overflow u64".into() },
    )?;
    let admitted = declaration
        .masks(mask_bytes)
        .design_expansion(design_bytes)
        .learner_fold_copies(0, 0)
        .solver_scratch(workspace)
        .retained_scores_draws(0)
        .output(8)
        .workers(1)
        .admit(RouteBudget::new(execution_byte_budget(ctx)))
        .map_err(|error| CausalError::Resource { message: error.to_string() })?;
    ctx.acquire_admission(1, admitted.total()).map_err(admission_error)
}

fn execution_byte_budget(ctx: &ExecutionContext) -> u64 {
    let pool = ctx.admission().byte_budget();
    match ctx.memory.hard_limit_bytes {
        Some(limit) => limit.min(pool),
        None => pool,
    }
}

fn admission_error(error: AdmissionError) -> CausalError {
    match error {
        AdmissionError::Cancelled => cancelled(),
        other => CausalError::Resource { message: other.to_string() },
    }
}

fn cancelled() -> CausalError {
    CausalError::Cancelled { stage: super::super::super::stage::STAGE_ESTIMATE_POINT }
}

fn compile_error(message: &str) -> CausalError {
    CausalError::Compile { message: message.to_owned() }
}

impl LoweredLinearGaussianRoute {
    /// Treatment, outcome, and adjustment columns, read through the design layout.
    ///
    /// A weighted problem keeps the weighted design prepared with the program.
    /// The layout is still built, so the source columns are on this path.
    fn seam_problem(&self) -> Result<PreparedEstimationProblem, CausalError> {
        let lowering = self.program.lowering();
        let treatment = lowering.treatment;
        let outcome = lowering.outcome;
        let adjustment = Arc::clone(&lowering.adjustment);
        let mut ids = Vec::with_capacity(adjustment.len().saturating_add(2));
        ids.push(treatment);
        ids.push(outcome);
        ids.extend_from_slice(&adjustment);
        let mask = self.data.complete_case_mask(&ids)?;
        let mut rows = Vec::new();
        let mut selected = Vec::new();
        for (index, keep) in mask.into_iter().enumerate() {
            if !keep {
                continue;
            }
            let raw = u64::try_from(index)
                .map_err(|_| compile_error("row index does not fit in u64"))?;
            rows.push(GlobalId::new(raw));
            selected.push(index);
        }
        let mut views = Vec::with_capacity(ids.len());
        for id in &ids {
            views.push(self.data.column(*id)?);
        }
        let layout = DesignLayout::from_views(&views, &rows)
            .map_err(|error| compile_error(&error.to_string()))?;
        if self.program.problem().sampling_weights.is_some() {
            return Ok(self.program.problem().clone());
        }
        let treatment_column =
            layout.column(0).ok_or_else(|| compile_error("design layout omitted treatment"))?;
        let outcome_column =
            layout.column(1).ok_or_else(|| compile_error("design layout omitted the outcome"))?;
        let mut covariates = Vec::with_capacity(adjustment.len());
        for (offset, id) in adjustment.iter().enumerate() {
            let column = layout
                .column(offset.saturating_add(2))
                .ok_or_else(|| compile_error("design layout omitted an adjustment column"))?;
            covariates.push((*id, column));
        }
        let design = CompiledDesign::linear_adjustment(
            treatment_column,
            &covariates,
            outcome_column,
            &selected,
        )?;
        let mut problem = self.program.problem().clone();
        problem.design = design;
        problem.treatment = Arc::from(treatment_column);
        Ok(problem)
    }
}

fn analysis_wire(
    query: CausalQueryWire,
    identification: IdentificationResultWire,
    estimate: f64,
    standard_error: f64,
) -> AnalysisResultWire {
    let published_error =
        (standard_error.is_finite() && standard_error >= 0.0).then_some(standard_error);
    AnalysisResultWire {
        query,
        identification,
        identification_variables: None,
        temporal_identification: Vec::new(),
        estimate: Some(estimate),
        policy_value: None,
        continuous_dose_response: None,
        panel_did: None,
        synthetic_control: None,
        synthetic_did: None,
        local_polynomial_ratio: None,
        randomized_effect: None,
        survival: None,
        posterior_row: None,
        longitudinal_regime: None,
        sequential_contrast: None,
        history_match: None,
        interference_inference: None,
        interventional_distribution: None,
        standard_error: published_error,
        interval_lower: None,
        interval_upper: None,
        assumptions: Vec::new(),
        diagnostics: Vec::new(),
        refutations: Vec::new(),
        response: None,
        posterior_artifact: None,
        mediation_grid: None,
        structural_response: None,
        unit_effects: None,
        cate: None,
        fitted_effect: None,
        cate_se: None,
        cate_leaf_dispersion: None,
        outcome_oof_r2: None,
        treatment_oof_logloss: None,
        crossfit_folds: None,
        crossfit_seed: None,
        learner_provenance: Vec::new(),
    }
}

struct Admitted<'a> {
    data: &'a TabularData,
    query: &'a AverageEffectQuery,
    graph: &'a Dag,
    identifier: IdentifierId,
}

impl Study {
    /// Lower the static ordinary-least-squares intervention route.
    ///
    /// The study must resolve to `linear.adjustment.ate` on a static DAG average
    /// contrast with an empty modifier list. Population, pair, and identification
    /// refusals are the ones the checked estimator already returns.
    /// [`Self::compile`] builds the physical plan and does not identify again.
    /// The retained program is what [`LoweredLinearGaussianRoute::execute`] fits.
    ///
    /// # Errors
    ///
    /// The study is a different route, the fit is not ordinary least squares, or
    /// identification or preparation refuses.
    pub(crate) fn lower_linear_gaussian_intervention(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<LoweredLinearGaussianRoute, CausalError> {
        let admitted = self.admit_linear_gaussian()?;
        let logical =
            self.compile_logical_ate_dag(admitted.data, admitted.query, admitted.graph)?;
        let physical = logical.compile_physical(ctx)?;
        let identification = self.identify_linear_gaussian(&admitted, ctx)?;
        let (program, fitter, point_then_bootstrap) =
            prepare_linear_gaussian(self, admitted.data, &identification)?;
        let product = retain_linear_gaussian(&identification, &program)?;
        let resample = fitter.bootstrap_replicates > 0;
        let data = admitted.data.clone();
        Ok(LoweredLinearGaussianRoute {
            scientific: linear_gaussian_scientific_problem(
                admitted.query,
                self.inference.is_bayesian(),
            )?,
            identification: product,
            identified: identification,
            data,
            program,
            fitter,
            point_then_bootstrap,
            evidence: LinearGaussianEvidenceProgram {
                support_status: self.support_status,
                operators: linear_gaussian_program_operators()?,
                decision: LinearGaussianDecision::ContrastMean,
            },
            physical_operators: linear_gaussian_physical_operators(&physical, resample)?,
            physical,
            recompile: RefCell::new(PreparedRecompile::after_lower()),
            cached_claim: RefCell::new(None),
        })
    }

    fn admit_linear_gaussian(&self) -> Result<Admitted<'_>, CausalError> {
        let (identifier, estimator) = self.resolve_static_ids();
        if estimator != EstimatorId::LinearAdjustmentAte {
            return Err(CausalError::Compile {
                message: format!(
                    "linear/Gaussian lowering admits estimator \"{}\"; got \"{}\"",
                    EstimatorId::LinearAdjustmentAte.as_str(),
                    estimator.as_str()
                ),
            });
        }
        let super::super::RouteInput::StaticAteDag { data, query, graph } = self.route_input()?
        else {
            return Err(CausalError::Compile {
                message: "linear/Gaussian lowering admits a static DAG average contrast".into(),
            });
        };
        if !query.effect_modifiers.is_empty() {
            return Err(CausalError::Compile {
                message: "linear/Gaussian lowering admits an unconditional average contrast".into(),
            });
        }
        Ok(Admitted { data, query, graph, identifier })
    }

    fn identify_linear_gaussian(
        &self,
        admitted: &Admitted<'_>,
        ctx: &ExecutionContext,
    ) -> Result<IdentificationResult, CausalError> {
        let rd = self.rd.map(|config| {
            antecedent_identify::SharpRdConfig::new(
                config.running_variable,
                config.cutoff,
                config.bandwidth,
            )
        });
        let (identification, _, _) = super::super::identification_from_cache_or(
            ctx,
            self.identification_cache.as_deref(),
            || {
                let identification = identify_static_query_with_rd(
                    admitted.identifier,
                    admitted.graph,
                    &CausalQuery::AverageEffect(admitted.query.clone()),
                    rd,
                )?;
                select_claim(identification, EstimatorId::LinearAdjustmentAte)
            },
        )?;
        Ok(identification)
    }
}

fn prepare_linear_gaussian(
    study: &Study,
    data: &TabularData,
    identification: &IdentificationResult,
) -> Result<(CheckedLinearAdjustmentAte, LinearAdjustmentAte, bool), CausalError> {
    let (fitter, point_then_bootstrap) = gaussian_fitter(study)?;
    let program = fitter.prepare_checked(data, identification, 0).map_err(CausalError::from)?;
    Ok((program, fitter, point_then_bootstrap))
}

fn gaussian_fitter(study: &Study) -> Result<(LinearAdjustmentAte, bool), CausalError> {
    match study.estimator_spec.as_deref() {
        None | Some(EstimatorSpec::Default(EstimatorId::LinearAdjustmentAte)) => {
            let mut fitter = LinearAdjustmentAte::new();
            fitter.bootstrap_replicates = study.bootstrap_replicates;
            Ok((fitter, true))
        }
        Some(EstimatorSpec::LinearAdjustmentAte(cfg)) => {
            if !matches!(cfg.fit_kind, LinearFitKind::Ols) {
                return Err(CausalError::Compile {
                    message: "linear/Gaussian lowering admits ordinary least squares".into(),
                });
            }
            Ok(((**cfg).clone(), false))
        }
        Some(other) => Err(CausalError::Compile {
            message: format!(
                "linear/Gaussian lowering admits estimator \"{}\"; got \"{}\"",
                EstimatorId::LinearAdjustmentAte.as_str(),
                other.id().as_str()
            ),
        }),
    }
}

fn retain_linear_gaussian(
    identification: &IdentificationResult,
    program: &CheckedLinearAdjustmentAte,
) -> Result<CompiledIdentificationProduct, CausalError> {
    if program.required_assumptions() != &identification.required_assumptions {
        return Err(CausalError::Compile {
            message: "checked linear program dropped an identification assumption".into(),
        });
    }
    let unresolved = unresolved_obligations(&identification.required_assumptions);
    let factors = required_factors(program);
    let (outcome, claim_class) = identification_claim(identification.status);
    let worlds = world_obligations(outcome, claim_class, &unresolved);
    let assembled =
        AssembledFunctionalObligation::new(outcome, claim_class, unresolved.clone(), None);
    let mut parts = CompiledProductParts::new(outcome, claim_class);
    parts.required_factors = factors;
    parts.unresolved_obligations = unresolved;
    parts.world_obligations = worlds;
    parts.assembled = Some(assembled);
    CompiledIdentificationProduct::retain(identification, parts)
        .map_err(|refusal| CausalError::Compile { message: refusal.to_string() })
}

fn required_factors(program: &CheckedLinearAdjustmentAte) -> Vec<RequiredFactor> {
    program
        .factor_requirements()
        .iter()
        .enumerate()
        .map(|(index, factor)| {
            RequiredFactor::new(format!("factor.{index}"), Arc::clone(&factor.variables))
        })
        .collect()
}

fn world_obligations(
    outcome: IdentificationOutcome,
    claim_class: IdentificationClaimClass,
    unresolved: &[UnresolvedObligation],
) -> Vec<WorldObligation> {
    [CONTROL_WORLD, ACTIVE_WORLD]
        .into_iter()
        .map(|raw| {
            WorldObligation::new(
                antecedent_core::ScientificWorldId::from_raw(raw),
                outcome,
                claim_class,
                unresolved.to_vec(),
                None,
            )
        })
        .collect()
}

fn identification_claim(
    status: antecedent_core::IdentificationStatus,
) -> (IdentificationOutcome, IdentificationClaimClass) {
    match status {
        antecedent_core::IdentificationStatus::NonparametricallyIdentified
        | antecedent_core::IdentificationStatus::IdentifiedUnderParametricRestrictions
        | antecedent_core::IdentificationStatus::IdentifiedUnderPriorRestrictions => {
            (IdentificationOutcome::Identified, IdentificationClaimClass::IdentifiedClaim)
        }
        antecedent_core::IdentificationStatus::PartiallyIdentified
        | antecedent_core::IdentificationStatus::GraphDependent
        | antecedent_core::IdentificationStatus::NotIdentified => {
            (IdentificationOutcome::Unidentified, IdentificationClaimClass::IdentifiedClaim)
        }
    }
}

fn unresolved_obligations(assumptions: &AssumptionSet) -> Vec<UnresolvedObligation> {
    assumptions
        .entries
        .iter()
        .enumerate()
        .filter(|(_, record)| {
            matches!(record.status, AssumptionStatus::Declared | AssumptionStatus::Contradicted)
        })
        .map(|(index, record)| {
            UnresolvedObligation::new(
                obligation_code(&record.assumption, index),
                obligation_detail(record),
            )
        })
        .collect()
}

fn obligation_code(assumption: &Assumption, index: usize) -> String {
    let stem = match assumption {
        Assumption::CausalMarkov => "causal_markov",
        Assumption::Faithfulness => "faithfulness",
        Assumption::CausalSufficiency => "causal_sufficiency",
        Assumption::Consistency => "consistency",
        Assumption::Positivity => "positivity",
        Assumption::NoInterference => "no_interference",
        Assumption::Stationarity => "stationarity",
        Assumption::PiecewiseStationarity => "piecewise_stationarity",
        Assumption::NoSelectionBias => "no_selection_bias",
        Assumption::ExclusionRestriction { .. } => "exclusion_restriction",
        Assumption::Monotonicity => "monotonicity",
        Assumption::ParametricRestriction(_) => "parametric_restriction",
        Assumption::PriorRestriction(_) => "prior_restriction",
        Assumption::Custom { .. } => "custom",
    };
    format!("{stem}.{index}")
}

fn obligation_detail(record: &AssumptionRecord) -> String {
    match &record.assumption {
        Assumption::ParametricRestriction(restriction) => {
            format!("{}: {}", record.status.as_str(), restriction.id)
        }
        Assumption::PriorRestriction(restriction) => {
            format!("{}: {}", record.status.as_str(), restriction.id)
        }
        Assumption::Custom { id, .. } => format!("{}: {id}", record.status.as_str()),
        _ => record.status.as_str().to_string(),
    }
}

#[cfg(test)]
mod tests {
    use antecedent_core::{
        AverageEffectQuery, CausalSchemaBuilder, ExecutionContext, TargetForm, TargetPopulation,
    };
    use antecedent_data::TabularData;
    use antecedent_estimate::{LinearAdjustmentAte, LinearFitKind};
    use antecedent_graph::Dag;
    use antecedent_identify::IdentificationOutcome;

    use antecedent_stats::fit_counts::count_least_squares_solves;

    use crate::RefuteSuite;
    use crate::analysis::prepared::recompile::{LifecycleInput, LifecycleProduct};
    use crate::strategy_table::EstimatorId;

    use super::super::super::Study;

    fn contrast_study() -> Study {
        let schema = CausalSchemaBuilder::new()
            .continuous("t")
            .treatment()
            .continuous("y")
            .outcome()
            .build()
            .unwrap();
        let treatment = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
        let outcome = [1.0, 3.0, 1.0, 3.0, 1.0, 3.0, 1.0, 3.0];
        let data = TabularData::try_from_schema_f64(
            schema.clone(),
            [("t", treatment.as_slice()), ("y", outcome.as_slice())],
        )
        .unwrap();
        let graph = Dag::from_named_edges(&schema, &[("t", "y")]).unwrap();
        let query =
            AverageEffectQuery::binary_ate(schema.id_of("t").unwrap(), schema.id_of("y").unwrap());
        Study::tabular(data)
            .graph(graph)
            .query(query)
            .refute(RefuteSuite::None)
            .bootstrap_replicates(0)
            .build()
            .unwrap()
    }

    #[test]
    #[allow(clippy::float_cmp, reason = "the retained fit must match Study::run bit for bit")]
    fn linear_gaussian_route_executes_without_rebuilding_a_builder() {
        let study = contrast_study();
        let ctx = ExecutionContext::for_tests(1);
        let route = study.lower_linear_gaussian_intervention(&ctx).unwrap();
        let expected = study.run(&ctx).unwrap().effect();
        drop(study);
        let claim = route.execute(&ctx).unwrap();
        assert_eq!(claim.estimate.ate, expected);
        assert_eq!(claim.receipt.bootstrap_replicates_requested, Some(0));
        assert!(matches!(route.scientific().form(), TargetForm::Contrast(_)));
        let again = route.execute(&ctx).unwrap();
        assert_eq!(claim.estimate.se_analytic(), again.estimate.se_analytic());
        assert_ne!(route.program().executable_functional(), route.program().source_functional());
        assert_eq!(route.evidence().decision(), super::LinearGaussianDecision::ContrastMean);
    }

    #[test]
    fn linear_gaussian_route_keeps_support_obligations() {
        let study = contrast_study();
        let ctx = ExecutionContext::for_tests(1);
        let route = study.lower_linear_gaussian_intervention(&ctx).unwrap();
        assert_eq!(route.evidence().support_status(), study.support_status());
        assert_eq!(route.identification().assumptions(), route.program().required_assumptions());
        assert!(route.identification().is_identified_claim());
        assert_eq!(route.identification().outcome(), IdentificationOutcome::Identified);
        assert_eq!(route.identification().world_obligations().len(), 2);
        assert!(
            route
                .identification()
                .assembled_obligation()
                .is_some_and(|obligation| { obligation.is_identified_claim() })
        );
        let claim = route.execute(&ctx).unwrap();
        assert_eq!(&claim.estimate.provenance.assumptions, route.identification().assumptions());
        assert_eq!(claim.support_status, study.support_status());
    }

    #[test]
    fn linear_gaussian_route_preserves_physical_plan() {
        let study = contrast_study();
        let ctx = ExecutionContext::for_tests(1);
        let route = study.lower_linear_gaussian_intervention(&ctx).unwrap();
        let compiled = study.compile(&ctx).unwrap();
        assert_eq!(route.physical().record, compiled.record);
        let effects: Vec<_> =
            route.physical_operators().iter().map(|op| op.effect().as_str()).collect();
        assert_eq!(effects, ["read", "fit", "intervene"]);
        let graphs: Vec<_> =
            route.evidence().operators().iter().map(|op| op.graph().as_str()).collect();
        assert_eq!(graphs, ["provenance", "execution", "causal"]);
    }

    #[test]
    fn linear_gaussian_route_refuses_a_different_estimator() {
        let study = contrast_study();
        let ctx = ExecutionContext::for_tests(1);
        assert!(study.compile(&ctx).is_ok());
        let refused = rebuild_like(&study, true, |builder| {
            builder.estimator(EstimatorId::PropensityWeighting)
        });
        let err = refused.lower_linear_gaussian_intervention(&ctx).unwrap_err();
        let message = err.to_string();
        assert!(message.contains("linear.adjustment.ate"), "{message}");
    }

    #[test]
    fn linear_gaussian_route_refuses_another_target_population() {
        let base = contrast_study();
        let ctx = ExecutionContext::for_tests(1);
        let study = rebuild_like(&base, true, |builder| {
            let query = match base.query.clone() {
                antecedent_core::CausalQuery::AverageEffect(query) => {
                    query.with_target_population(TargetPopulation::Treated)
                }
                other => panic!("expected an average effect, got {other:?}"),
            };
            builder.query(query)
        });
        let err = study.lower_linear_gaussian_intervention(&ctx).unwrap_err();
        let message = err.to_string();
        assert_eq!(
            antecedent_core::reason_code::split_prefix(&message).map(|(code, _)| code),
            Some("population_not_estimable"),
            "{message}"
        );
    }

    #[test]
    fn linear_gaussian_route_refuses_non_ols_fit() {
        let base = contrast_study();
        let ctx = ExecutionContext::for_tests(1);
        let ridge = LinearAdjustmentAte::new().with_fit_kind(LinearFitKind::Ridge { lambda: 1.0 });
        let study = rebuild_like(&base, false, |builder| builder.estimator(ridge));
        let err = study.lower_linear_gaussian_intervention(&ctx).unwrap_err();
        assert!(err.to_string().contains("ordinary least squares"), "{err}");
    }

    #[test]
    fn prepared_linear_gaussian_study_executes_the_retained_program() {
        let study = contrast_study();
        let ctx = ExecutionContext::for_tests(1);
        let route = study.lower_linear_gaussian_intervention(&ctx).unwrap();
        let compiled = study.compile(&ctx).unwrap();
        assert_eq!(compiled.record, route.physical().record);
        assert_eq!(compiled.logical.record, route.physical().logical.record);
        let prepared = study.prepare(&ctx).unwrap();
        assert_eq!(prepared.plan().record, route.physical().record);
        let sealed = prepared.checked_linear_adjustment().unwrap();
        assert_eq!(sealed.executable_functional(), route.program().executable_functional());
        assert_eq!(sealed.source_functional(), route.program().source_functional());
        assert_eq!(sealed.required_assumptions(), route.program().required_assumptions());
        let expected = study.run(&ctx).unwrap().effect();
        let claim = route.execute(&ctx).unwrap();
        assert_eq!(claim.estimate.ate.to_bits(), expected.to_bits());
        assert_eq!(&claim.estimate.provenance.assumptions, route.program().required_assumptions(),);
        let retained = prepared.estimate_retained(&ctx).unwrap();
        assert_eq!(retained.effect().to_bits(), claim.estimate.ate.to_bits());
    }

    #[test]
    #[allow(clippy::float_cmp, reason = "a reused fit must return the same contrast bits")]
    fn linear_gaussian_utility_and_action_changes_perform_zero_causal_fits() {
        let (route, ctx, ate, fits_before) = fitted_route();
        for input in [LifecycleInput::Utility, LifecycleInput::Action] {
            route.apply_input(input);
            assert!(route.product_is_valid(LifecycleProduct::CausalFit), "{input:?}");
            assert!(route.product_is_valid(LifecycleProduct::Structural), "{input:?}");
            assert!(route.product_is_valid(LifecycleProduct::Unrelated), "{input:?}");
            assert!(!route.product_is_valid(LifecycleProduct::Decision), "{input:?}");
            let (claim, solves) = count_least_squares_solves(|| route.execute(&ctx).unwrap());
            assert_eq!(solves, 0, "{input:?}");
            assert_eq!(route.causal_fit_count(), fits_before, "{input:?}");
            assert_eq!(claim.estimate.ate, ate, "{input:?}");
            assert!(route.product_is_valid(LifecycleProduct::Decision), "{input:?}");
        }
    }

    #[test]
    fn linear_gaussian_prior_change_preserves_only_valid_structure() {
        let (route, ctx, _, fits_before) = fitted_route();
        route.apply_input(LifecycleInput::Prior);
        assert!(route.product_is_valid(LifecycleProduct::Structural));
        assert!(route.product_is_valid(LifecycleProduct::Unrelated));
        assert!(!route.product_is_valid(LifecycleProduct::PriorStructural));
        assert!(!route.product_is_valid(LifecycleProduct::CausalFit));
        assert!(!route.product_is_valid(LifecycleProduct::Decision));
        let (_, solves) = count_least_squares_solves(|| route.execute(&ctx).unwrap());
        assert!(solves > 0);
        assert_eq!(route.causal_fit_count(), fits_before + 1);
    }

    #[test]
    fn linear_gaussian_source_period_and_history_changes_keep_unrelated_products() {
        for input in [LifecycleInput::Source, LifecycleInput::Period, LifecycleInput::History] {
            let (route, ctx, _, _) = fitted_route();
            route.apply_input(input);
            assert!(!route.product_is_valid(LifecycleProduct::CausalFit), "{input:?}");
            assert!(!route.product_is_valid(LifecycleProduct::Decision), "{input:?}");
            assert!(route.product_is_valid(LifecycleProduct::Structural), "{input:?}");
            assert!(route.product_is_valid(LifecycleProduct::PriorStructural), "{input:?}");
            assert!(route.product_is_valid(LifecycleProduct::Unrelated), "{input:?}");
            let (_, solves) = count_least_squares_solves(|| route.execute(&ctx).unwrap());
            assert!(solves > 0, "{input:?}");
            assert!(route.product_is_valid(LifecycleProduct::Structural), "{input:?}");
            assert!(route.product_is_valid(LifecycleProduct::PriorStructural), "{input:?}");
            assert!(route.product_is_valid(LifecycleProduct::Unrelated), "{input:?}");
        }
    }

    fn fitted_route() -> (super::LoweredLinearGaussianRoute, ExecutionContext, f64, u64) {
        let study = contrast_study();
        let ctx = ExecutionContext::for_tests(1);
        let route = study.lower_linear_gaussian_intervention(&ctx).unwrap();
        let compiled = study.compile(&ctx).unwrap();
        assert_eq!(compiled.record, route.physical().record);
        assert_eq!(compiled.logical.record, route.physical().logical.record);
        let (claim, solves) = count_least_squares_solves(|| route.execute(&ctx).unwrap());
        assert!(solves > 0);
        let fits = route.causal_fit_count();
        assert_eq!(fits, 1);
        (route, ctx, claim.estimate.ate, fits)
    }

    fn rebuild_like(
        study: &Study,
        zero_bootstrap: bool,
        edit: impl FnOnce(crate::analysis::StudyBuilder) -> crate::analysis::StudyBuilder,
    ) -> Study {
        let antecedent_core::CausalQuery::AverageEffect(query) = study.query.clone() else {
            panic!("expected an average effect");
        };
        let data = match &study.data {
            crate::analysis::builder::DataInput::Tabular(data) => data.clone(),
            _ => panic!("expected tabular data"),
        };
        let graph = study.graph.as_dag().unwrap().clone();
        let mut builder = Study::tabular(data).graph(graph).query(query).refute(RefuteSuite::None);
        if zero_bootstrap {
            builder = builder.bootstrap_replicates(0);
        }
        edit(builder).build().unwrap()
    }
}
