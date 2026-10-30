//! Linear adjustment ATE estimator.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![cfg_attr(
    test,
    allow(
        clippy::cast_possible_truncation,
        reason = "test fixtures compare exact constants and index with small literals"
    )
)]

use std::sync::Arc;

use antecedent_core::{
    AssumptionSet, AverageEffectQuery, ExecutionContext, Intervention, TargetPopulation, VariableId,
};
use antecedent_data::materialize::{
    PreparationBudget, PreparationControls, PreparationProduct, PreparationReceipt, StreamingScan,
    prepare_table,
};
use antecedent_data::{Contrast, MissingPolicy, TableView, TabularData};
use antecedent_expr::{
    CausalExprArena, DomainRef, ExprId, ExprNode, FactorRequirement, FunctionalProgram,
    IdentifiedEstimand, OutcomeExprId, ProgramLimits, ProgramSchema, ProgramVariable,
};
use antecedent_identify::{IdentificationResult, IdentificationStatus};
use antecedent_stats::{
    CompiledDesign, DenseLinearAlgebra, FaerBackend, LassoOptions, LeastSquaresWorkspace,
    MEstimateOptions, RidgeIntercept, fit_huber_m, fit_lasso_with_ones_column, fit_ridge_with,
    predict_lasso, ridge_gram_inverse,
};

use crate::arm::TreatmentArm;
pub use crate::effect_estimate::{
    AnalyticSe, BlockResampling, BlockSeRecord, BootstrapAccounting, CandidateSelectionRecord,
    CrossfitRecord, EffectEstimate, EstimateDiagnostics, EstimateHeterogeneity, EstimateProvenance,
    EstimateScores, EstimateUncertainty, FamilyInference, JointRecord, ScenarioBand,
};
use crate::error::EstimationError;
use crate::inference::InferenceSpec;
use crate::overlap::OverlapPolicy;
use crate::prepare::{
    CheckedSelection, require_adjustment_shaped, require_unique_roles, treatment_contrast,
    validate_ate_query_with_targets,
};
use crate::se::{AnalyticSeKind, SeSpec, residual_sandwich_coef_se, score_sandwich_coef_se};

mod bootstrap;
mod checked;
mod solve;

pub use checked::{CheckedAdjustmentLowering, CheckedLinearAdjustmentAte};
pub(crate) use solve::intervention_f64;

use solve::*;

/// Prepared estimation problem (compiled design retained).
#[derive(Clone, Debug)]
pub struct PreparedEstimationProblem {
    /// Compiled design.
    pub design: CompiledDesign,
    /// Estimand method tag.
    pub method: Arc<str>,
    /// Adjustment set.
    pub adjustment_set: Arc<[VariableId]>,
    /// Overlap policy applied.
    pub overlap: OverlapPolicy,
    /// Active − control treatment contrast used for the ATE scaling.
    pub treatment_delta: f64,
    /// Target population (ATT/ATC use g-computation over the arm’s covariate law).
    pub target_population: TargetPopulation,
    /// Complete-case treatment values (aligned with design rows).
    pub treatment: Arc<[f64]>,
    /// Active treatment level.
    pub active: f64,
    /// Control treatment level.
    pub control: f64,
    /// The two contrasted levels and the rows excluded, when the treatment is a discrete column
    /// with further levels and the restriction to the two levels was declared.
    pub treatment_levels: Option<crate::propensity::DeclaredTreatmentLevels>,
    /// What declared sampling weights did to this problem. When present the design matrix and
    /// outcome are the weighted-least-squares transform (every row multiplied by the square
    /// root of its mean-one weight), so each normal-equation and sandwich formula downstream
    /// reads the sampling-weighted score `w_i x_i e_i`; `treatment` stays unweighted.
    pub sampling_weights: Option<crate::sampling_weights::SamplingWeightRecord>,
    /// Preparation receipt for the finite table this problem was read from.
    pub preparation: PreparationReceipt,
}

/// Estimation workspace (reusable across bootstrap replicates).
#[derive(Clone, Debug, Default)]
pub struct EstimationWorkspace {
    /// OLS scratch.
    pub ols: LeastSquaresWorkspace,
}

/// Linear fit family for [`LinearAdjustmentAte`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LinearFitKind {
    /// Ordinary least squares.
    Ols,
    /// Ridge with penalty `lambda` (intercept unpenalized when constant).
    ///
    /// The treatment coefficient is penalized, so the point estimate is shrunk and an
    /// interval around it has no bias term: no analytic SE is published (`se_analytic` is
    /// NaN); use bootstrap. The exported influence function is the ridge one.
    Ridge {
        /// Ridge penalty λ.
        lambda: f64,
    },
    /// Lasso with penalty `lambda`.
    ///
    /// Analytic SE is permanently omitted: classical / active-set sandwich SEs are
    /// invalid after selection, and debiased Lasso changes the point estimator.
    /// Use bootstrap (`bootstrap_replicates > 0`); `se_analytic` is NaN.
    Lasso {
        /// Lasso penalty λ.
        lambda: f64,
    },
    /// Huber M-estimation with tuning constant `c`.
    ///
    /// Analytic SE is the M-estimator's sandwich (Huber 1981), not the OLS formula on Huber
    /// residuals. The residual scale is `1.4826·MAD`, or `√(π/2)·` the mean absolute
    /// deviation about the median when the MAD is exactly zero (a discrete or
    /// zero-inflated outcome, where a zero scale would down-weight every non-majority
    /// row); either way it is treated as known, and an exact fit has scale zero with unit
    /// weights. The estimand is the Huber M-functional at that scale, not the mean.
    Huber {
        /// Huber tuning constant (default 1.345).
        c: f64,
    },
}

impl Default for LinearFitKind {
    fn default() -> Self {
        Self::Ols
    }
}

/// Linear adjustment estimator for backdoor ATE.
#[derive(Clone, Debug)]
pub struct LinearAdjustmentAte {
    /// Backend.
    pub backend: FaerBackend,
    /// Inference declarations: bootstrap size and interval request, analytic standard-error
    /// kind, cluster / multiway / panel labels and the sampling design of the rows.
    pub inference: InferenceSpec,
    /// Overlap policy (must be explicit in).
    pub overlap: OverlapPolicy,
    /// Linear fit family (default OLS).
    pub fit_kind: LinearFitKind,
    /// Registry for named [`TargetPopulation::Predicate`] selections.
    pub population_registry: Option<antecedent_core::PopulationRegistry>,
    /// Sampling weights of every row of the source table. The fit is weighted least squares
    /// with the linearization (sandwich) variance; see [`crate::sampling_weights`] for the
    /// estimand, normalization and zero-weight rules. Only the OLS fit reads them.
    pub sampling_weights: Option<crate::sampling_weights::SamplingWeights>,
    /// Byte budget for preparation. `None` admits the resident table's payload.
    pub preparation_budget: Option<PreparationBudget>,
    /// Explicit contrasts for categorical adjustment columns, in request order.
    pub contrasts: Vec<(VariableId, Contrast)>,
    /// Policy for rows at neither contrasted level when the treatment is a discrete column with
    /// further levels (see [`crate::OtherLevels`]). The default refuses: the treatment enters
    /// the regression as one numeric column, so a column such as `0/1/2` would be read as a
    /// dose and "1 against 0" would be its slope, not the contrast of the two levels.
    pub other_levels: crate::propensity::OtherLevels,
    /// Read the treatment as a numeric dose: the contrast is the fitted slope times
    /// `active - control`, and a discrete column with further levels is not a refusal. The
    /// default is `false`, which keeps the two-level contrast rule above. Set by the
    /// diagnostics of a dose-response design, whose original estimate is a dose slope.
    pub dose_reading: bool,
}

impl Default for LinearAdjustmentAte {
    fn default() -> Self {
        Self::new()
    }
}

impl LinearAdjustmentAte {
    /// Default: 200 bootstrap replicates, explicit overlap override, OLS.
    #[must_use]
    pub fn new() -> Self {
        Self {
            backend: FaerBackend,
            inference: InferenceSpec::new(200),
            overlap: OverlapPolicy::ExplicitOverride,
            fit_kind: LinearFitKind::Ols,
            population_registry: None,
            sampling_weights: None,
            preparation_budget: None,
            contrasts: Vec::new(),
            other_levels: crate::propensity::OtherLevels::default(),
            dose_reading: false,
        }
    }

    /// Read the treatment as a numeric dose (see [`Self::dose_reading`]).
    #[must_use]
    pub const fn with_dose_reading(mut self, dose_reading: bool) -> Self {
        self.dose_reading = dose_reading;
        self
    }

    /// Declare what to do with rows at neither contrasted level when the treatment is a
    /// discrete column with further levels. The default refuses; the exclusion restricts the
    /// target population to the rows at the two levels and is reported on the result.
    #[must_use]
    pub const fn with_other_levels(mut self, other_levels: crate::propensity::OtherLevels) -> Self {
        self.other_levels = other_levels;
        self
    }

    crate::inference::with_bootstrap_interval!(inference);

    /// Admit preparation only when the resident payload fits `budget`.
    #[must_use]
    pub const fn with_preparation_budget(mut self, budget: PreparationBudget) -> Self {
        self.preparation_budget = Some(budget);
        self
    }

    /// Expand `id` with `contrast` during preparation.
    #[must_use]
    pub fn with_contrast(mut self, id: VariableId, contrast: Contrast) -> Self {
        self.contrasts.push((id, contrast));
        self
    }

    fn refuse_unweighted_routes(&self) -> Result<(), EstimationError> {
        if self.sampling_weights.is_some() && !matches!(self.fit_kind, LinearFitKind::Ols) {
            return Err(crate::sampling_weights::unsupported_route(
                "sampling weights are read by the ordinary least-squares fit only; ridge, lasso \
                 and Huber fits have no sampling-weighted variance",
            ));
        }
        if self.sampling_weights.is_some()
            && matches!(self.inference.se_kind, AnalyticSeKind::Homoskedastic)
        {
            return Err(crate::sampling_weights::unsupported_route(
                "the classical homoskedastic standard error treats weights as precision and is \
                 invalid for sampling weights; request a heteroskedasticity-robust kind \
                 (hc0..hc3) or the cluster sandwich",
            ));
        }
        Ok(())
    }

    /// Refuse `scan`. Linear adjustment reads a finite table.
    ///
    /// Materialize the frame before calling [`Self::prepare`].
    #[must_use = "the refusal is the decision not to read the scan"]
    pub fn reject_stream(_scan: &StreamingScan) -> EstimationError {
        EstimationError::unsupported_coded(
            antecedent_core::reason_code!("route_not_supported"),
            "linear adjustment cannot stream; materialize the frame or refuse it",
        )
    }

    crate::inference::with_bootstrap_replicates!(inference);

    /// Set the overlap policy. `prepare` requires [`OverlapPolicy::ExplicitOverride`] since
    /// this is a regression (not propensity-based) path.
    #[must_use]
    pub const fn with_overlap(mut self, overlap: OverlapPolicy) -> Self {
        self.overlap = overlap;
        self
    }

    crate::inference::with_se_kind!(inference);

    crate::inference::with_cluster_ids!(inference);

    crate::inference::with_source_cluster_ids!(inference);

    crate::inference::with_sampling_design!(inference);

    /// Declare sampling weights for every row of the source table (OLS only). The classical
    /// homoskedastic SE kind is refused with weights at `prepare`: a model-based variance
    /// treats weights as precision, which is invalid for sampling weights. Choose an HC or
    /// cluster kind.
    #[must_use]
    pub fn with_sampling_weights(
        mut self,
        weights: crate::sampling_weights::SamplingWeights,
    ) -> Self {
        self.sampling_weights = Some(weights);
        self
    }

    crate::inference::with_multiway_ids!(inference);

    crate::inference::with_panel_times!(inference);

    /// Set the linear fit family (default [`LinearFitKind::Ols`]).
    ///
    /// **Lasso trap**: with `LinearFitKind::Lasso { .. }`, the analytic SE is permanently
    /// `NaN` — classical / active-set sandwich SEs are invalid after selection, and debiased
    /// Lasso changes the point estimator. Pair this with a nonzero
    /// [`Self::with_bootstrap_replicates`] to get a usable SE; this setter stays dumb and
    /// does not enforce that pairing.
    #[must_use]
    pub const fn with_fit_kind(mut self, fit_kind: LinearFitKind) -> Self {
        self.fit_kind = fit_kind;
        self
    }

    /// Prepare design from tabular data, identified estimand, and query levels.
    ///
    /// # Errors
    ///
    /// Missing columns, unsupported query options, type errors, or overlap policy not set.
    pub fn prepare(
        &self,
        data: &TabularData,
        estimand: &IdentifiedEstimand,
        query: &AverageEffectQuery,
    ) -> Result<PreparedEstimationProblem, EstimationError> {
        self.prepare_unchecked(data, estimand, query)
    }

    fn prepare_unchecked(
        &self,
        data: &TabularData,
        estimand: &IdentifiedEstimand,
        query: &AverageEffectQuery,
    ) -> Result<PreparedEstimationProblem, EstimationError> {
        self.prepare_with_backend(
            data,
            estimand,
            query,
            antecedent_core::physical_choice::PhysicalAlternative::DirectPreparedNumericViews,
        )
    }

    fn prepare_with_backend(
        &self,
        data: &TabularData,
        estimand: &IdentifiedEstimand,
        query: &AverageEffectQuery,
        backend: antecedent_core::physical_choice::PhysicalAlternative,
    ) -> Result<PreparedEstimationProblem, EstimationError> {
        use antecedent_core::physical_choice::PhysicalAlternative;
        let assemble = match backend {
            PhysicalAlternative::DirectPreparedNumericViews => {
                crate::prepare::direct_adjustment_design
            }
            PhysicalAlternative::MaterializedDesign => {
                crate::prepare::materialized_adjustment_design
            }
            _ => {
                return Err(EstimationError::unsupported_coded(
                    antecedent_core::reason_code!("route_not_supported"),
                    "linear adjustment requires retained rows; streamed and lazy preparation are unavailable",
                ));
            }
        };
        crate::util::require_explicit_override(
            self.overlap,
            "LinearAdjustmentAte requires ExplicitOverride overlap policy",
        )?;
        require_adjustment_shaped(
            estimand,
            "LinearAdjustmentAte expects an adjustment-shaped estimand",
        )?;
        self.inference.require_declared_design_read("LinearAdjustmentAte")?;
        validate_ate_query_with_targets(query)?;
        let treatment = query.treatment;
        let outcome = query.outcome;
        let (active, control, treatment_delta) = treatment_contrast(&query.active, &query.control)?;
        if let Some(source) = self.inference.source_cluster_ids.as_deref() {
            crate::se::require_source_labels(
                source,
                self.inference.cluster_ids.as_deref(),
                data.row_count(),
            )?;
        }

        let mut ids = Vec::with_capacity(2 + estimand.adjustment_set.len());
        ids.push(treatment);
        ids.push(outcome);
        ids.extend_from_slice(&estimand.adjustment_set);
        let mut controls = PreparationControls::finite_resident();
        controls.budget = self.preparation_budget;
        let PreparationProduct { expanded, mut row_mask, receipt: preparation } =
            prepare_table(data, &ids, MissingPolicy::CompleteCase, &self.contrasts, controls)?;
        crate::prepare::intersect_predicate_mask(
            &mut row_mask,
            &query.target_population,
            data.row_count(),
            self.population_registry.as_ref(),
        )?;
        let treatment_levels = if self.dose_reading {
            None
        } else {
            crate::prepare::restrict_to_contrast_levels(
                data,
                treatment,
                &mut row_mask,
                (active, control),
                self.other_levels,
            )?
        };
        self.refuse_unweighted_routes()?;
        let retained = self
            .sampling_weights
            .as_ref()
            .map(|weights| weights.restrict(&mut row_mask))
            .transpose()?;
        refuse_non_finite_cells(data, treatment, outcome, &estimand.adjustment_set, &row_mask)?;
        let design =
            assemble(data, treatment, outcome, &estimand.adjustment_set, &expanded, &row_mask)?;
        debug_assert!(
            crate::design_width::quote_adjustment_design(
                data,
                &estimand.adjustment_set,
                &self.contrasts
            )
            .ok()
            .is_none_or(|quote| quote.columns == design.ncols),
            "the planner's quoted design width must equal the built design"
        );
        let t: Arc<[f64]> = Arc::from(&design.matrix[design.nrows..2 * design.nrows]);
        let design = match &retained {
            Some(retained) => {
                crate::sampling_weights::root_weight_design(design, &retained.weights)
            }
            None => design,
        };
        Ok(PreparedEstimationProblem {
            design,
            method: Arc::clone(&estimand.method),
            adjustment_set: Arc::clone(&estimand.adjustment_set),
            overlap: self.overlap,
            treatment_delta,
            target_population: query.target_population.clone(),
            treatment: t,
            active,
            control,
            treatment_levels,
            sampling_weights: retained.map(|retained| retained.record),
            preparation,
        })
    }

    /// Prepare linear adjustment from one selected, identified average-effect claim.
    ///
    /// This path verifies the selected claim status, its adjustment roles, and that the
    /// existing expression root is exactly the back-door ATE functional implied by the
    /// query arms and selected adjustment set. It retains the source root and constructs
    /// a separate observational executable root. This checked lowering currently supports
    /// `AllObserved`, whose averaging law is represented by the empirical `P(Z)` factor.
    /// It retains the original arena and both roots in the returned receipt. The legacy
    /// [`Self::prepare`] method remains available for callers that manage this context
    /// themselves and does not make the checked-execution guarantee.
    ///
    /// # Errors
    ///
    /// The selected claim is absent, not nonparametrically identified, incompatible with
    /// linear adjustment, or its functional does not match its query and role structure.
    #[allow(
        clippy::too_many_lines,
        reason = "the checked receipt is built atomically from the validated target, query, schema, and lowering"
    )]
    pub fn prepare_checked(
        &self,
        data: &TabularData,
        identification: &IdentificationResult,
        estimand_index: usize,
    ) -> Result<CheckedLinearAdjustmentAte, EstimationError> {
        self.prepare_checked_with_backend(
            data,
            identification,
            estimand_index,
            antecedent_core::physical_choice::PhysicalAlternative::DirectPreparedNumericViews,
        )
    }

    /// Prepare a checked claim using a verified physical design implementation.
    ///
    /// # Errors
    /// Refuses unsupported backends and the same invalid claims as [`Self::prepare_checked`].
    #[allow(
        clippy::too_many_lines,
        reason = "validate the complete checked claim before binding its physical design"
    )]
    pub fn prepare_checked_with_backend(
        &self,
        data: &TabularData,
        identification: &IdentificationResult,
        estimand_index: usize,
        backend: antecedent_core::physical_choice::PhysicalAlternative,
    ) -> Result<CheckedLinearAdjustmentAte, EstimationError> {
        let CheckedSelection { estimand: target, claim } =
            CheckedSelection::resolve(identification, estimand_index)?;
        if claim.status != IdentificationStatus::NonparametricallyIdentified {
            return Err(EstimationError::data_msg(format!(
                "linear adjustment requires a nonparametrically identified claim; got {}",
                claim.status.as_str()
            )));
        }
        let query = identification.average_effect().ok_or_else(|| {
            EstimationError::data_msg(
                "linear adjustment requires an average-effect identification query",
            )
        })?;
        require_adjustment_shaped(
            target,
            "LinearAdjustmentAte expects an adjustment-shaped estimand",
        )?;
        if !target.instruments.is_empty() || !target.mediators.is_empty() {
            return Err(EstimationError::data_msg(
                "adjustment estimand carries incompatible instrument or mediator roles",
            ));
        }
        require_unique_roles(
            &target.adjustment_set,
            &[query.treatment, query.outcome],
            "adjustment roles must be unique and exclude treatment and outcome",
        )?;
        if query.outcome_functional != antecedent_core::OutcomeFunctional::Mean {
            return Err(EstimationError::unsupported_coded(
                antecedent_core::reason_code!("route_not_supported"),
                "checked linear adjustment currently supports the mean outcome functional",
            ));
        }
        if query.target_population != TargetPopulation::AllObserved {
            return Err(EstimationError::unsupported_coded(
                antecedent_core::reason_code!("route_not_supported"),
                "checked linear adjustment lowering currently supports AllObserved population averaging",
            ));
        }

        let (active, control, _) = treatment_contrast(&query.active, &query.control)?;
        let mut checked_arena = identification.arena.clone();
        let expected = {
            checked_arena.backdoor_ate(
                query.treatment,
                query.outcome,
                &target.adjustment_set,
                antecedent_core::Value::f64(active),
                antecedent_core::Value::f64(control),
            )
        };
        if expected != target.functional {
            return Err(EstimationError::data_msg(
                "selected functional does not match the query contrast, treatment arms, and adjustment roles",
            ));
        }

        let executable = observational_adjustment_arm(
            &mut checked_arena,
            query.treatment,
            query.outcome,
            &target.adjustment_set,
        );

        let program_schema = ProgramSchema::new(
            data.schema()
                .variables()
                .iter()
                .map(|v| (v.id, ProgramVariable { name: Arc::clone(&v.name) })),
        );
        let program = FunctionalProgram::new(
            checked_arena.clone(),
            program_schema,
            target.functional,
            executable,
            ProgramLimits::default(),
        )?;
        let factor_requirements: Arc<[FactorRequirement]> =
            Arc::from(program.factor_requirements().to_vec());

        let problem = self.prepare_with_backend(data, target, query, backend)?;
        Ok(CheckedLinearAdjustmentAte {
            target: target.clone(),
            source_functional: target.functional,
            executable_functional: executable,
            lowering: CheckedAdjustmentLowering {
                source: target.functional,
                executable,
                treatment: query.treatment,
                outcome: query.outcome,
                adjustment: Arc::clone(&target.adjustment_set),
                active,
                control,
                population: query.target_population.clone(),
            },
            program,
            factor_requirements,
            query: query.clone(),
            required_assumptions: claim.required_assumptions.clone(),
            problem,
            backend,
        })
    }

    /// Rebuild the physical design for another dataset with the same semantic schema.
    ///
    /// The checked program and lowering are reused unchanged. This verifies the data schema,
    /// retained query, target roles, and source/executable correspondence before preparing
    /// the new row and design binding.
    #[allow(
        clippy::float_cmp,
        reason = "treatment arms are categorical design values and must match the retained query exactly"
    )]
    pub fn rebind_checked(
        &self,
        checked: &CheckedLinearAdjustmentAte,
        data: &TabularData,
    ) -> Result<CheckedLinearAdjustmentAte, EstimationError> {
        self.rebind_checked_with_backend(checked, data, checked.backend)
    }

    /// Rebind retained semantics using a newly selected executable design backend.
    ///
    /// # Errors
    /// Refuses unsupported backends or a changed semantic schema or claim.
    #[allow(clippy::float_cmp, reason = "retained treatment arms must match exactly")]
    pub fn rebind_checked_with_backend(
        &self,
        checked: &CheckedLinearAdjustmentAte,
        data: &TabularData,
        backend: antecedent_core::physical_choice::PhysicalAlternative,
    ) -> Result<CheckedLinearAdjustmentAte, EstimationError> {
        let schema = ProgramSchema::new(
            data.schema()
                .variables()
                .iter()
                .map(|v| (v.id, ProgramVariable { name: Arc::clone(&v.name) })),
        );
        let mapping = checked.program.mapping();
        if checked.program.schema() != &schema
            || checked.target.functional != checked.source_functional
            || mapping.source != checked.source_functional
            || mapping.executable != checked.executable_functional
            || checked.lowering.source != checked.source_functional
            || checked.lowering.executable != checked.executable_functional
            || checked.lowering.treatment != checked.query.treatment
            || checked.lowering.outcome != checked.query.outcome
            || checked.lowering.adjustment != checked.target.adjustment_set
            || checked.lowering.population != checked.query.target_population
            || checked.query.outcome_functional != antecedent_core::OutcomeFunctional::Mean
            || checked.query.target_population != TargetPopulation::AllObserved
        {
            return Err(EstimationError::data_msg(
                "checked adjustment program, query, roles, or data semantic schema no longer agree",
            ));
        }
        let (active, control, _) =
            treatment_contrast(&checked.query.active, &checked.query.control)?;
        if active != checked.lowering.active || control != checked.lowering.control {
            return Err(EstimationError::data_msg(
                "checked adjustment arm bindings no longer agree with the retained query",
            ));
        }
        let mut rebound = checked.clone();
        rebound.problem =
            self.prepare_with_backend(data, &checked.target, &checked.query, backend)?;
        rebound.backend = backend;
        Ok(rebound)
    }

    /// Fit a checked preparation using the assumptions retained with its selected claim.
    ///
    /// # Errors
    ///
    /// Fit, uncertainty, or cooperative execution failure.
    pub fn fit_checked(
        &self,
        checked: &CheckedLinearAdjustmentAte,
        workspace: &mut EstimationWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<EffectEstimate, EstimationError> {
        self.fit(&checked.problem, workspace, ctx, checked.required_assumptions.clone())
    }

    /// Fit ATE with optional IID bootstrap.
    ///
    /// # Errors
    ///
    /// Fit / SE failure.
    pub fn fit(
        &self,
        problem: &PreparedEstimationProblem,
        workspace: &mut EstimationWorkspace,
        ctx: &ExecutionContext,
        assumptions: AssumptionSet,
    ) -> Result<EffectEstimate, EstimationError> {
        let point = self.fit_point(problem, workspace, assumptions)?;
        self.attach_bootstrap(problem, workspace, ctx, point)
    }

    /// Point estimate + analytic SE only (no bootstrap).
    ///
    /// # Errors
    ///
    /// Fit / SE failure.
    pub fn fit_point(
        &self,
        problem: &PreparedEstimationProblem,
        workspace: &mut EstimationWorkspace,
        assumptions: AssumptionSet,
    ) -> Result<EffectEstimate, EstimationError> {
        self.fit_point_with_ols_residuals(problem, workspace, assumptions).map(|(point, _)| point)
    }

    /// [`Self::fit_point`] that also hands back the OLS residuals of the fit it ran, so a
    /// caller that needs the normal-equation scores does not solve the regression again.
    /// `None` when the configured fit is not ordinary least squares (ridge, lasso and Huber
    /// residuals are not the OLS normal-equation residuals).
    ///
    /// # Errors
    ///
    /// Fit / SE failure.
    pub fn fit_point_with_ols_residuals(
        &self,
        problem: &PreparedEstimationProblem,
        workspace: &mut EstimationWorkspace,
        assumptions: AssumptionSet,
    ) -> Result<(EffectEstimate, Option<Vec<f64>>), EstimationError> {
        antecedent_core::execution_attempt::run_operation(
            antecedent_core::execution_attempt::Operation::AdjustedFit,
            || self.fit_point_with_ols_residuals_unobserved(problem, workspace, assumptions),
        )
    }

    fn fit_point_with_ols_residuals_unobserved(
        &self,
        problem: &PreparedEstimationProblem,
        workspace: &mut EstimationWorkspace,
        assumptions: AssumptionSet,
    ) -> Result<(EffectEstimate, Option<Vec<f64>>), EstimationError> {
        let mut fit = self.fit_coefficients(problem, workspace)?;
        let t_col = problem
            .design
            .treatment_column()
            .ok_or_else(|| EstimationError::stats_msg("missing treatment column"))?;
        let ate = gcomp_or_coef_ate(problem, &fit.coefficients, t_col)?;
        let nrows = problem.design.nrows;
        let (se_coef, influence) = self.coefficient_se_and_influence(problem, &fit, t_col)?;
        let se_analytic = se_coef * problem.treatment_delta.abs();

        let mut point = EffectEstimate::new(ate, se_analytic, assumptions, problem.overlap)
            .with_n_obs(u64::try_from(nrows).unwrap_or(u64::MAX))
            .with_se_kind(self.inference.se_kind)
            .with_sampling_weights(problem.sampling_weights)
            .with_preparation(Some(crate::prepare::preparation_summary(&problem.preparation)))
            .with_influence(influence.map(Arc::from));
        if let Some(levels) = problem.treatment_levels {
            point = point.with_treatment_levels(levels);
        }
        let residuals =
            matches!(fit.variance, FitVariance::Ols).then(|| std::mem::take(&mut fit.residuals));
        Ok((point, residuals))
    }

    /// Treatment-coefficient SE and influence function of the estimator that actually ran.
    ///
    /// OLS: `RSS/(n−p)·[(XᵀX)⁻¹]_tt` or the residual sandwich. Huber: the M-estimator's
    /// sandwich, bread `Σ ψ′ x xᵀ` and scores `s·ψ(r/s)`. Ridge and Lasso publish no analytic
    /// SE (shrinkage bias is not in a sampling variance, and post-selection inference is
    /// invalid); their influence function is the ridge one, resp. none, never the OLS one.
    fn coefficient_se_and_influence(
        &self,
        problem: &PreparedEstimationProblem,
        fit: &LinearFit,
        t_col: usize,
    ) -> Result<(f64, Option<Vec<f64>>), EstimationError> {
        let x = &problem.design.matrix;
        let nrows = problem.design.nrows;
        let ncols = problem.design.ncols;
        let n = nrows as f64;
        let p = ncols as f64;
        let delta = problem.treatment_delta;
        let retained_clusters = self
            .inference
            .source_cluster_ids
            .as_deref()
            .map(|source| crate::se::retained_row_labels(source, &problem.design.row_selection))
            .transpose()?;
        let cluster_ids = retained_clusters.as_deref().or(self.inference.cluster_ids.as_deref());
        match &fit.variance {
            FitVariance::Ols => {
                // `(XᵀX)⁻¹` once: the homoskedastic SE reads its treatment diagonal and
                // the influence function its treatment row.
                let xtx_inv = crate::util::xtx_inverse(x, nrows, ncols);
                let se_coef = if nrows <= ncols {
                    f64::NAN
                } else if let Some(se) = residual_sandwich_coef_se(
                    SeSpec {
                        kind: self.inference.se_kind,
                        cluster_ids,
                        multiway_ids: self.inference.multiway_ids.as_deref(),
                        panel_times: self.inference.panel_times.as_deref(),
                    },
                    x,
                    nrows,
                    ncols,
                    &fit.residuals,
                    t_col,
                )? {
                    se
                } else {
                    let sigma2 = fit.rss / (n - p);
                    xtx_inv.as_ref().map_or(f64::NAN, |inv| {
                        (sigma2 * inv[t_col * ncols + t_col].max(0.0)).sqrt()
                    })
                };
                let influence = treatment_coef_influence(
                    x,
                    nrows,
                    ncols,
                    t_col,
                    &fit.residuals,
                    delta,
                    xtx_inv.as_deref(),
                );
                Ok((se_coef, Some(influence)))
            }
            FitVariance::Ridge { lambda } => {
                let bread = ridge_gram_inverse(x, nrows, ncols, *lambda)?;
                let influence = treatment_coef_influence(
                    x,
                    nrows,
                    ncols,
                    t_col,
                    &fit.residuals,
                    delta,
                    bread.as_deref(),
                );
                Ok((f64::NAN, Some(influence)))
            }
            FitVariance::Lasso => Ok((f64::NAN, None)),
            FitVariance::Huber { scores, curvature } => {
                let bread = curvature_gram_inverse(x, nrows, ncols, curvature);
                let mean_curvature = curvature.iter().sum::<f64>() / n;
                let se_coef = if let Some(se) = score_sandwich_coef_se(
                    SeSpec {
                        kind: self.inference.se_kind,
                        cluster_ids,
                        multiway_ids: self.inference.multiway_ids.as_deref(),
                        panel_times: self.inference.panel_times.as_deref(),
                    },
                    x,
                    nrows,
                    ncols,
                    scores,
                    curvature,
                    t_col,
                )? {
                    se
                } else if mean_curvature > 0.0 && n > p {
                    // Huber (1981, §7.10): Var(β̂) = κ² · [Σ s²ψ²/(n−p)] / m² · (XᵀX)⁻¹ with
                    // m = mean ψ′ and κ = 1 + (p/n) var(ψ′)/m². For an indicator ψ′,
                    // var(ψ′) = m(1−m).
                    huber_homoskedastic_se(x, nrows, ncols, t_col, scores, mean_curvature)
                } else {
                    f64::NAN
                };
                let influence = treatment_coef_influence(
                    x,
                    nrows,
                    ncols,
                    t_col,
                    scores,
                    delta,
                    bread.as_deref(),
                );
                Ok((se_coef, Some(influence)))
            }
        }
    }

    fn fit_coefficients(
        &self,
        problem: &PreparedEstimationProblem,
        workspace: &mut EstimationWorkspace,
    ) -> Result<LinearFit, EstimationError> {
        let solved = if matches!(self.fit_kind, LinearFitKind::Ols) {
            let fit = problem
                .design
                .fit_ols(&self.backend, &mut workspace.ols)
                .map_err(EstimationError::from)?;
            SolvedLinear {
                coefficients: fit.coefficients,
                residuals: fit.residuals,
                rss: fit.rss,
                huber_scale: None,
            }
        } else {
            solve_linear(
                self.fit_kind,
                self.backend,
                &problem.design.matrix,
                problem.design.nrows,
                problem.design.ncols,
                &problem.design.outcome,
                &mut workspace.ols,
            )?
        };
        linear_fit_from_solved(self.fit_kind, solved)
    }
}

impl crate::estimator::Estimator<TabularData> for LinearAdjustmentAte {
    type Fit = EffectEstimate;

    fn prepare(
        &self,
        data: &TabularData,
        estimand: &IdentifiedEstimand,
        query: &AverageEffectQuery,
        _ctx: &ExecutionContext,
    ) -> Result<PreparedEstimationProblem, EstimationError> {
        self.prepare_unchecked(data, estimand, query)
    }

    fn fit(
        &self,
        problem: &PreparedEstimationProblem,
        workspace: &mut EstimationWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<Self::Fit, EstimationError> {
        Self::fit(self, problem, workspace, ctx, AssumptionSet::new())
    }
}

/// Lower one arm of the checked back-door target to its observational g-formula.
fn observational_adjustment_arm(
    arena: &mut CausalExprArena,
    treatment: VariableId,
    outcome: VariableId,
    adjustment: &[VariableId],
) -> ExprId {
    let z = arena.intern_var_set(adjustment.iter().copied());
    let y = arena.intern_var_set([outcome]);
    let t_and_z =
        arena.intern_var_set(std::iter::once(treatment).chain(adjustment.iter().copied()));
    let empty = arena.empty_var_set();
    let empty_intervention = arena.empty_intervention_set();
    let marginal_z =
        arena.intern_distribution(z, empty, empty_intervention, DomainRef::Observational);
    // Treatment remains an ordinary conditioning coordinate. The typed lowering binds it
    // at evaluation time through Assignment; the provider receives no intervention request.
    let outcome_given_tz =
        arena.intern_distribution(y, t_and_z, empty_intervention, DomainRef::Observational);
    let factors = arena.intern_list([outcome_given_tz, marginal_z]);
    let product = arena.intern(ExprNode::Product(factors));
    let averaged = arena.intern(ExprNode::SumOut { variables: z, expr: product });
    arena.intern(ExprNode::Expectation {
        function: OutcomeExprId::identity(outcome),
        distribution: averaged,
    })
}

/// Refuse an infinite value in a retained row of the treatment, the outcome or an adjustment
/// column. A missing cell is the missing-data policy's business; an infinite one is not a
/// missing value and would otherwise surface as a rank-deficient design.
fn refuse_non_finite_cells(
    data: &TabularData,
    treatment: VariableId,
    outcome: VariableId,
    adjustment: &[VariableId],
    row_mask: &[bool],
) -> Result<(), EstimationError> {
    use antecedent_data::ColumnView;
    for id in [treatment, outcome].into_iter().chain(adjustment.iter().copied()) {
        let ColumnView::Float64(column) = data.column(id)? else { continue };
        let values = column.values.as_slice();
        let Some(row) =
            values.iter().zip(row_mask).position(|(value, &kept)| kept && value.is_infinite())
        else {
            continue;
        };
        let name = data
            .schema()
            .get(id)
            .map_or_else(|_| format!("#{}", id.raw()), |variable| variable.name.to_string());
        return Err(EstimationError::refused(
            antecedent_core::reason_code!("invalid_argument"),
            format!(
                "column `{name}` holds a non-finite value ({}) at row {row}, which no linear \
                 adjustment can read. Next attempt: remove or correct the row, or declare the \
                 value missing so a missing-data policy handles it",
                values[row]
            ),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests;
