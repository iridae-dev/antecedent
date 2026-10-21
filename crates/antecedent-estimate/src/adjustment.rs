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
use antecedent_data::categorical::ExpandedContrast;
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
    MEstimateOptions, fit_huber_m, fit_lasso_with_ones_column, fit_ridge, predict_lasso,
    ridge_gram_inverse,
};

pub use crate::effect_estimate::{
    AnalyticSe, BlockResampling, BlockSeRecord, BootstrapAccounting, CandidateSelectionRecord,
    CrossfitRecord, EffectEstimate, EstimateDiagnostics, EstimateHeterogeneity, EstimateProvenance,
    EstimateScores, EstimateUncertainty, FamilyInference, JointRecord, ScenarioBand,
};
use crate::error::EstimationError;
use crate::overlap::OverlapPolicy;
use crate::prepare::{
    CheckedSelection, require_adjustment_shaped, require_unique_roles, treatment_contrast,
    validate_ate_query_with_targets,
};
use crate::se::{AnalyticSeKind, SeSpec, residual_sandwich_coef_se, score_sandwich_coef_se};

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
    /// What declared sampling weights did to this problem. When present the design matrix and
    /// outcome are the weighted-least-squares transform (every row multiplied by the square
    /// root of its mean-one weight), so each normal-equation and sandwich formula downstream
    /// reads the sampling-weighted score `w_i x_i e_i`; `treatment` stays unweighted.
    pub sampling_weights: Option<crate::sampling_weights::SamplingWeightRecord>,
    /// Preparation receipt for the finite table this problem was read from.
    pub preparation: PreparationReceipt,
}

/// Receipt for a linear adjustment preparation checked against an identification result.
///
/// The retained arena and target make the proof context inspectable after the builder has
/// been discarded. `problem` is the physical linear design prepared from that same query.
#[derive(Clone, Debug)]
pub struct CheckedLinearAdjustmentAte {
    /// Selected causal target and its estimator roles.
    target: IdentifiedEstimand,
    /// Selected source functional root.
    source_functional: ExprId,
    /// One-arm observational adjustment functional derived from the source target.
    /// Evaluate with treatment bound to each declared arm, then subtract control from active.
    executable_functional: ExprId,
    /// Checked role and population receipt for the source-to-executable lowering.
    lowering: CheckedAdjustmentLowering,
    /// Structurally checked owner of the source and executable roots.
    program: FunctionalProgram,
    /// Provider factors discovered while checking the executable root.
    factor_requirements: Arc<[FactorRequirement]>,
    /// Query that was checked and used to bind the source target.
    query: AverageEffectQuery,
    /// Assumptions attached to the selected identification claim.
    required_assumptions: AssumptionSet,
    /// Checked and prepared linear adjustment design.
    problem: PreparedEstimationProblem,
}

impl CheckedLinearAdjustmentAte {
    /// Selected causal target.
    #[must_use]
    pub fn target(&self) -> &IdentifiedEstimand {
        &self.target
    }
    /// Checked program that owns the target and observational execution root.
    #[must_use]
    pub fn program(&self) -> &FunctionalProgram {
        &self.program
    }
    /// Checked source root.
    #[must_use]
    pub fn source_functional(&self) -> ExprId {
        self.source_functional
    }
    /// Checked executable arm root.
    #[must_use]
    pub fn executable_functional(&self) -> ExprId {
        self.executable_functional
    }
    /// Typed source-to-executable lowering.
    #[must_use]
    pub fn lowering(&self) -> &CheckedAdjustmentLowering {
        &self.lowering
    }
    /// Checked provider factor requirements.
    #[must_use]
    pub fn factor_requirements(&self) -> &[FactorRequirement] {
        &self.factor_requirements
    }
    /// Selected identification assumptions.
    #[must_use]
    pub fn required_assumptions(&self) -> &AssumptionSet {
        &self.required_assumptions
    }
    /// Prepared numerical design.
    #[must_use]
    pub fn problem(&self) -> &PreparedEstimationProblem {
        &self.problem
    }
}

/// Checked semantic correspondence for a static adjustment ATE lowering.
///
/// The executable root is an observational one-arm functional with treatment free as a
/// conditioning coordinate. Consumers evaluate it twice using the retained active and
/// control values and subtract control from active.
#[derive(Clone, Debug, PartialEq)]
pub struct CheckedAdjustmentLowering {
    /// Source interventional target root.
    pub source: ExprId,
    /// Executable observational root.
    pub executable: ExprId,
    /// Treatment role used by both arms.
    pub treatment: VariableId,
    /// Outcome role.
    pub outcome: VariableId,
    /// Adjustment roles in semantic order.
    pub adjustment: Arc<[VariableId]>,
    /// Active intervention value.
    pub active: f64,
    /// Control intervention value.
    pub control: f64,
    /// Population averaging semantics implemented by this checked lowering.
    pub population: TargetPopulation,
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
    /// Bootstrap replicates (0 = skip bootstrap).
    pub bootstrap_replicates: u32,
    /// Overlap policy (must be explicit in).
    pub overlap: OverlapPolicy,
    /// Analytic SE estimator (default homoskedastic).
    pub se_kind: AnalyticSeKind,
    /// Optional cluster ids (length = prepared `nrows`) for cluster / panel SE.
    pub cluster_ids: Option<Vec<u32>>,
    /// Optional cluster labels of every row of the source table (length = source rows),
    /// subset to the retained rows by the design's row selection. Exclusive with
    /// [`Self::cluster_ids`].
    pub source_cluster_ids: Option<Vec<u32>>,
    /// Optional multiway cluster ids for [`AnalyticSeKind::Multiway`].
    pub multiway_ids: Option<Vec<Vec<u32>>>,
    /// Optional panel time labels (length = prepared `nrows`) for panel HAC.
    pub panel_times: Option<Vec<i64>>,
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
            bootstrap_replicates: 200,
            overlap: OverlapPolicy::ExplicitOverride,
            se_kind: AnalyticSeKind::Homoskedastic,
            cluster_ids: None,
            source_cluster_ids: None,
            multiway_ids: None,
            panel_times: None,
            fit_kind: LinearFitKind::Ols,
            population_registry: None,
            sampling_weights: None,
            preparation_budget: None,
            contrasts: Vec::new(),
        }
    }

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
        if self.sampling_weights.is_some() && matches!(self.se_kind, AnalyticSeKind::Homoskedastic)
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
        EstimationError::unsupported(
            "linear adjustment cannot stream; materialize the frame or refuse it",
        )
    }

    /// Set the number of bootstrap replicates used for the bootstrap standard error.
    ///
    /// Defaults to 200. Set to `0` to skip bootstrapping and report only the analytic SE
    /// (`NaN` when [`Self::with_fit_kind`] is [`LinearFitKind::Lasso`] or
    /// [`LinearFitKind::Ridge`]).
    #[must_use]
    pub const fn with_bootstrap_replicates(mut self, replicates: u32) -> Self {
        self.bootstrap_replicates = replicates;
        self
    }

    /// Set the overlap policy. `prepare` requires [`OverlapPolicy::ExplicitOverride`] since
    /// this is a regression (not propensity-based) path.
    #[must_use]
    pub const fn with_overlap(mut self, overlap: OverlapPolicy) -> Self {
        self.overlap = overlap;
        self
    }

    /// Set the analytic standard-error kind (default [`AnalyticSeKind::Homoskedastic`]).
    #[must_use]
    pub const fn with_se_kind(mut self, se_kind: AnalyticSeKind) -> Self {
        self.se_kind = se_kind;
        self
    }

    /// Set cluster ids (length must equal the prepared design's row count) for
    /// [`AnalyticSeKind::Cluster`] / panel SE.
    #[must_use]
    pub fn with_cluster_ids(mut self, cluster_ids: Vec<u32>) -> Self {
        self.cluster_ids = Some(cluster_ids);
        self
    }

    /// Set cluster labels for every row of the source table (length must equal its row
    /// count). Rows dropped as incomplete or outside the target population drop their
    /// labels with them, so the labels stay with their rows.
    #[must_use]
    pub fn with_source_cluster_ids(mut self, cluster_ids: Vec<u32>) -> Self {
        self.source_cluster_ids = Some(cluster_ids);
        self
    }

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

    /// Set multiway cluster ids (one `Vec<u32>` per clustering dimension) for
    /// [`AnalyticSeKind::Multiway`].
    #[must_use]
    pub fn with_multiway_ids(mut self, multiway_ids: Vec<Vec<u32>>) -> Self {
        self.multiway_ids = Some(multiway_ids);
        self
    }

    /// Set panel time labels (length must equal the prepared design's row count) for
    /// panel HAC standard errors.
    #[must_use]
    pub fn with_panel_times(mut self, panel_times: Vec<i64>) -> Self {
        self.panel_times = Some(panel_times);
        self
    }

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
        crate::util::require_explicit_override(
            self.overlap,
            "LinearAdjustmentAte requires ExplicitOverride overlap policy",
        )?;
        require_adjustment_shaped(
            estimand,
            "LinearAdjustmentAte expects an adjustment-shaped estimand",
        )?;
        validate_ate_query_with_targets(query)?;
        let treatment = query.treatment;
        let outcome = query.outcome;
        let (active, control, treatment_delta) = treatment_contrast(&query.active, &query.control)?;
        if let Some(source) = self.source_cluster_ids.as_deref() {
            crate::se::require_source_labels(
                source,
                self.cluster_ids.as_deref(),
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
        self.refuse_unweighted_routes()?;
        let retained = self
            .sampling_weights
            .as_ref()
            .map(|weights| weights.restrict(&mut row_mask))
            .transpose()?;
        let t = data.float64_masked(treatment, &row_mask).map_err(EstimationError::from)?;
        let y = data.float64_masked(outcome, &row_mask).map_err(EstimationError::from)?;
        let covs = adjustment_covariates(data, &estimand.adjustment_set, &expanded, &row_mask)?;
        let cov_refs: Vec<(VariableId, &[f64])> =
            covs.iter().map(|(id, v)| (*id, v.as_slice())).collect();
        let selected_rows: Vec<usize> =
            row_mask.iter().enumerate().filter_map(|(i, keep)| keep.then_some(i)).collect();
        let design = CompiledDesign::linear_adjustment(&t, &cov_refs, &y, &selected_rows)
            .map_err(EstimationError::from)?;
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
            treatment: Arc::from(t),
            active,
            control,
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
            return Err(EstimationError::unsupported(
                "checked linear adjustment currently supports the mean outcome functional",
            ));
        }
        if query.target_population != TargetPopulation::AllObserved {
            return Err(EstimationError::unsupported(
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
        )
        .map_err(|e| EstimationError::data_msg(format!("checked adjustment program: {e}")))?;
        let factor_requirements: Arc<[FactorRequirement]> =
            Arc::from(program.factor_requirements().to_vec());

        let problem = self.prepare_unchecked(data, target, query)?;
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
        rebound.problem = self.prepare_unchecked(data, &checked.target, &checked.query)?;
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
        let mut fit = self.fit_coefficients(problem, workspace)?;
        let t_col = problem
            .design
            .treatment_column()
            .ok_or_else(|| EstimationError::stats_msg("missing treatment column"))?;
        let ate = gcomp_or_coef_ate(problem, &fit.coefficients, t_col)?;
        let nrows = problem.design.nrows;
        let (se_coef, influence) = self.coefficient_se_and_influence(problem, &fit, t_col)?;
        let se_analytic = se_coef * problem.treatment_delta.abs();

        let point = EffectEstimate::new(ate, se_analytic, assumptions, problem.overlap)
            .with_n_obs(u64::try_from(nrows).unwrap_or(u64::MAX))
            .with_se_kind(self.se_kind)
            .with_sampling_weights(problem.sampling_weights)
            .with_preparation(Some(crate::prepare::preparation_summary(&problem.preparation)))
            .with_influence(influence.map(Arc::from));
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
            .source_cluster_ids
            .as_deref()
            .map(|source| crate::se::retained_row_labels(source, &problem.design.row_selection))
            .transpose()?;
        let cluster_ids = retained_clusters.as_deref().or(self.cluster_ids.as_deref());
        match &fit.variance {
            FitVariance::Ols => {
                // `(XᵀX)⁻¹` once: the homoskedastic SE reads its treatment diagonal and
                // the influence function its treatment row.
                let xtx_inv = crate::util::xtx_inverse(x, nrows, ncols);
                let se_coef = if let Some(se) = residual_sandwich_coef_se(
                    SeSpec {
                        kind: self.se_kind,
                        cluster_ids,
                        multiway_ids: self.multiway_ids.as_deref(),
                        panel_times: self.panel_times.as_deref(),
                    },
                    x,
                    nrows,
                    ncols,
                    &fit.residuals,
                    t_col,
                )? {
                    se
                } else {
                    let sigma2 = fit.rss / (n - p).max(1.0);
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
                        kind: self.se_kind,
                        cluster_ids,
                        multiway_ids: self.multiway_ids.as_deref(),
                        panel_times: self.panel_times.as_deref(),
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

    /// Attach bootstrap SE onto a point estimate (progressive uncertainty stage).
    ///
    /// # Errors
    ///
    /// Bootstrap failure.
    pub fn attach_bootstrap(
        &self,
        problem: &PreparedEstimationProblem,
        workspace: &mut EstimationWorkspace,
        ctx: &ExecutionContext,
        point: EffectEstimate,
    ) -> Result<EffectEstimate, EstimationError> {
        let boot = if self.bootstrap_replicates == 0 {
            None
        } else {
            let t_col = problem
                .design
                .treatment_column()
                .ok_or_else(|| EstimationError::stats_msg("missing treatment column"))?;
            Some(self.bootstrap_se(problem, workspace, ctx, t_col)?)
        };
        Ok(point.with_bootstrap(boot))
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

    /// ATE from the configured linear fit on a row-resample of a prepared design.
    ///
    /// `row_src[r]` copies design row `row_src[r]` onto output row `r`. Length must
    /// equal `problem.design.nrows`. For `AllObserved` / `Predicate` the ATE is
    /// `β_T · Δ` and does not reread treatment labels.
    ///
    /// # Errors
    ///
    /// Shape mismatch, out-of-range source row, or fit failure.
    pub fn ate_on_row_indices(
        &self,
        problem: &PreparedEstimationProblem,
        workspace: &mut EstimationWorkspace,
        row_src: &[usize],
    ) -> Result<f64, EstimationError> {
        let n = problem.design.nrows;
        let p = problem.design.ncols;
        if row_src.len() != n {
            return Err(EstimationError::data_msg(
                "row resample length must match the prepared design",
            ));
        }
        let mut x_boot = vec![0.0; n * p];
        let mut y_boot = vec![0.0; n];
        self.ate_on_row_indices_into(problem, workspace, row_src, &mut x_boot, &mut y_boot)
    }

    /// [`Self::ate_on_row_indices`] using caller-owned gather buffers.
    ///
    /// Here `row_src` may be shorter than the design (a sub-window resample); the
    /// buffers must hold `row_src.len()` rows (`x_boot`: `row_src.len() · ncols`).
    ///
    /// # Errors
    ///
    /// Shape mismatch, out-of-range source row, or fit failure.
    pub fn ate_on_row_indices_into(
        &self,
        problem: &PreparedEstimationProblem,
        workspace: &mut EstimationWorkspace,
        row_src: &[usize],
        x_boot: &mut [f64],
        y_boot: &mut [f64],
    ) -> Result<f64, EstimationError> {
        let t_col = problem
            .design
            .treatment_column()
            .ok_or_else(|| EstimationError::stats_msg("missing treatment column"))?;
        let coefs = self.fit_resampled_coefficients(problem, workspace, row_src, x_boot, y_boot)?;
        gcomp_or_coef_ate(problem, &coefs, t_col)
    }

    fn fit_resampled_coefficients(
        &self,
        problem: &PreparedEstimationProblem,
        workspace: &mut EstimationWorkspace,
        row_src: &[usize],
        x_boot: &mut [f64],
        y_boot: &mut [f64],
    ) -> Result<Vec<f64>, EstimationError> {
        let rows = problem.design.nrows;
        let p = problem.design.ncols;
        // Output rows follow the row map, which may be a sub-window of the design
        // (a multi-design shared time window); every source row must exist.
        let n = row_src.len();
        if y_boot.len() != n || x_boot.len() != n * p {
            return Err(EstimationError::data_msg("row resample buffers must match the row map"));
        }
        for (r, &src) in row_src.iter().enumerate() {
            if src >= rows {
                return Err(EstimationError::data_msg("row resample index out of range"));
            }
            y_boot[r] = problem.design.outcome[src];
            for c in 0..p {
                x_boot[c * n + r] = problem.design.matrix[c * rows + src];
            }
        }
        if self.fit_kind == LinearFitKind::Ols {
            return self
                .backend
                .least_squares_coefficients(x_boot, n, p, y_boot)
                .map_err(EstimationError::from);
        }
        Ok(solve_linear(self.fit_kind, self.backend, x_boot, n, p, y_boot, &mut workspace.ols)?
            .coefficients)
    }

    fn bootstrap_se(
        &self,
        problem: &PreparedEstimationProblem,
        _workspace: &mut EstimationWorkspace,
        ctx: &ExecutionContext,
        t_col: usize,
    ) -> Result<crate::util::BootstrapSeResult, EstimationError> {
        let n = problem.design.nrows;
        let p = problem.design.ncols;
        // Replicates own a cold workspace. The caller's scratch stays untouched: parallel
        // workers cannot share it, and a warm IRLS start would change the replicate fit.
        let resampling = (self.fit_kind == LinearFitKind::Ols && self.bootstrap_replicates >= 32)
            .then(|| {
                antecedent_stats::faer_backend::ResampledLeastSquares::prepare(
                    &problem.design.matrix,
                    n,
                    p,
                    &problem.design.outcome,
                    ctx,
                )
            })
            .flatten();
        crate::util::bootstrap_se_with_scratch(
            self.bootstrap_replicates,
            ctx,
            0xA7E_u64,
            n,
            || {
                (
                    EstimationWorkspace::default(),
                    Vec::new(),
                    Vec::new(),
                    antecedent_stats::faer_backend::ResampledLeastSquaresWorkspace::default(),
                )
            },
            |(ws, x_boot, y_boot, summary_ws), idx| {
                let candidate = match &resampling {
                    Some(prepared) => prepared.try_coefficients(idx, summary_ws),
                    None => Ok(None),
                };
                let coefficients = match candidate {
                    Ok(Some(coefs)) => Ok(coefs),
                    Ok(None) => {
                        x_boot.resize(n * p, 0.0);
                        y_boot.resize(n, 0.0);
                        self.fit_resampled_coefficients(problem, ws, idx, x_boot, y_boot)
                    }
                    Err(error) => Err(EstimationError::from(error)),
                };
                match coefficients {
                    Ok(coefs) => Ok(Some(gcomp_or_coef_ate(problem, &coefs, t_col)?)),
                    Err(error) if bootstrap_fit_failed(&error) => Ok(None),
                    Err(error) => Err(error),
                }
            },
        )
    }
}

/// Coefficients and raw residuals of one linear solve, before the variance model is attached.
struct SolvedLinear {
    coefficients: Vec<f64>,
    residuals: Vec<f64>,
    rss: f64,
    /// Huber IRLS scale. Absent for OLS, ridge, and lasso.
    huber_scale: Option<f64>,
}

/// One solver body for the point fit and for bootstrap resamples.
fn solve_linear(
    kind: LinearFitKind,
    backend: FaerBackend,
    x: &[f64],
    n: usize,
    p: usize,
    y: &[f64],
    workspace: &mut LeastSquaresWorkspace,
) -> Result<SolvedLinear, EstimationError> {
    match kind {
        LinearFitKind::Ols => {
            let fit =
                backend.least_squares(x, n, p, y, workspace).map_err(EstimationError::from)?;
            Ok(SolvedLinear {
                coefficients: fit.coefficients,
                residuals: fit.residuals,
                rss: fit.rss,
                huber_scale: None,
            })
        }
        LinearFitKind::Ridge { lambda } => {
            let fit = fit_ridge(x, n, p, y, lambda, &backend, workspace)
                .map_err(EstimationError::from)?;
            Ok(SolvedLinear {
                coefficients: fit.coefficients,
                residuals: fit.residuals,
                rss: fit.rss,
                huber_scale: None,
            })
        }
        LinearFitKind::Lasso { lambda } => {
            let fit = fit_lasso_with_ones_column(
                x,
                n,
                p,
                y,
                &LassoOptions { lambda, fit_intercept: true, ..LassoOptions::default() },
            )
            .map_err(EstimationError::from)?;
            // An unconverged replicate is a snapshot, not a draw from the estimator's
            // sampling distribution; the caller counts it as a failed replicate.
            require_converged(fit.converged, LASSO_UNCONVERGED)?;
            // Design keeps an all-ones column; stitch intercept back for g-computation.
            let mut coefficients = Vec::with_capacity(p);
            coefficients.push(fit.intercept);
            coefficients.extend_from_slice(&fit.coefficients);
            let pred = if fit.coefficients.is_empty() {
                vec![fit.intercept; n]
            } else {
                predict_lasso(&fit, &x[n..], n, p - 1).map_err(EstimationError::from)?
            };
            let mut residuals = vec![0.0; n];
            let mut rss = 0.0;
            for r in 0..n {
                let e = y[r] - pred[r];
                residuals[r] = e;
                rss += e * e;
            }
            Ok(SolvedLinear { coefficients, residuals, rss, huber_scale: None })
        }
        LinearFitKind::Huber { c } => {
            let opts = MEstimateOptions { c, ..MEstimateOptions::default() };
            let fit = fit_huber_m(x, n, p, y, &opts, &backend, workspace)
                .map_err(EstimationError::from)?;
            // An unconverged IRLS run is a snapshot of an unfinished reweighting
            // sequence, not the M-estimator's fixed point; publishing it — and an
            // analytic SE derived from it — would attach robust-estimand semantics
            // to a number that has none. Mirrors the GAM backfit gate in
            // `response.rs::fit_additive`.
            require_converged(fit.converged, HUBER_UNCONVERGED)?;
            let mut residuals = vec![0.0; n];
            let mut rss = 0.0;
            for r in 0..n {
                let mut pred = 0.0;
                for col in 0..p {
                    pred += x[col * n + r] * fit.coefficients[col];
                }
                let e = y[r] - pred;
                residuals[r] = e;
                rss += e * e;
            }
            Ok(SolvedLinear {
                coefficients: fit.coefficients,
                residuals,
                rss,
                huber_scale: Some(fit.scale),
            })
        }
    }
}

fn linear_fit_from_solved(
    kind: LinearFitKind,
    solved: SolvedLinear,
) -> Result<LinearFit, EstimationError> {
    let variance = match kind {
        LinearFitKind::Ols => FitVariance::Ols,
        LinearFitKind::Ridge { lambda } => FitVariance::Ridge { lambda },
        // Permanent policy: no analytic SE for Lasso (bootstrap only).
        LinearFitKind::Lasso { .. } => FitVariance::Lasso,
        LinearFitKind::Huber { c } => {
            // `solve_linear` records the IRLS scale on the Huber arm before returning.
            let scale = solved
                .huber_scale
                .ok_or_else(|| EstimationError::stats_msg("Huber fit is missing its IRLS scale"))?;
            let bound = c * scale;
            // Estimating equation `Σ x_i · s·ψ_c(r_i/s) = 0`: the score is the residual
            // clipped at `c·s`, and its derivative in the linear predictor is the
            // indicator of the unclipped region.
            let scores: Vec<f64> =
                solved.residuals.iter().map(|r| r.clamp(-bound, bound)).collect();
            let curvature: Vec<f64> =
                solved.residuals.iter().map(|r| f64::from(u8::from(r.abs() <= bound))).collect();
            FitVariance::Huber { scores, curvature }
        }
    };
    Ok(LinearFit {
        coefficients: solved.coefficients,
        residuals: solved.residuals,
        rss: solved.rss,
        variance,
    })
}

/// Fit failures and an empty target arm are failed replicates. A resample can be
/// singular, unconverged, or contain none of the target arm; those continue.
/// Shape errors in the caller's buffers propagate.
fn bootstrap_fit_failed(error: &EstimationError) -> bool {
    match error {
        EstimationError::Stats(_) => true,
        EstimationError::Unsupported { message }
            if *message == LASSO_UNCONVERGED || *message == HUBER_UNCONVERGED =>
        {
            true
        }
        EstimationError::Data(antecedent_data::DataError::InvalidArgument { message })
            if message == "target population left no rows for g-computation" =>
        {
            true
        }
        _ => false,
    }
}

/// G-computation of μ(active,Z)−μ(control,Z) averaged under the target arm’s covariate law.
///
/// Under a linear main-effects model this equals `β_T · Δ` for every target, including ATT/ATC.
fn gcomp_or_coef_ate(
    problem: &PreparedEstimationProblem,
    coefficients: &[f64],
    t_col: usize,
) -> Result<f64, EstimationError> {
    match problem.target_population {
        TargetPopulation::AllObserved | TargetPopulation::Predicate(_) => {
            Ok(coefficients[t_col] * problem.treatment_delta)
        }
        // The design is the weighted transform, which g-computation cannot average over. Under
        // the main-effects model every target has the effect `β_T · Δ`.
        TargetPopulation::Treated | TargetPopulation::Untreated
            if problem.sampling_weights.is_some() =>
        {
            Ok(coefficients[t_col] * problem.treatment_delta)
        }
        TargetPopulation::Treated | TargetPopulation::Untreated => {
            let n = problem.design.nrows;
            let ncols = problem.design.ncols;
            let want_treated = matches!(problem.target_population, TargetPopulation::Treated);
            let mut sum = 0.0;
            let mut count = 0usize;
            for r in 0..n {
                let treated = problem.treatment[r] > 0.5;
                if treated != want_treated {
                    continue;
                }
                let mut pred_a = 0.0;
                let mut pred_c = 0.0;
                for (c, &coef) in coefficients.iter().enumerate().take(ncols) {
                    let x = if c == t_col {
                        (problem.active, problem.control)
                    } else {
                        let v = problem.design.matrix[c * n + r];
                        (v, v)
                    };
                    pred_a += coef * x.0;
                    pred_c += coef * x.1;
                }
                sum += pred_a - pred_c;
                count += 1;
            }
            if count == 0 {
                return Err(EstimationError::data_msg(
                    "target population left no rows for g-computation",
                ));
            }
            Ok(sum / count as f64)
        }
        _ => Err(EstimationError::TargetPopulation),
    }
}

pub(crate) fn intervention_f64(intervention: &Intervention) -> Result<f64, EstimationError> {
    match intervention {
        Intervention::Set { value, .. } => value.as_f64().ok_or_else(|| {
            EstimationError::unsupported(" linear adjustment requires numeric treatment levels")
        }),
        _ => Err(EstimationError::unsupported(" linear adjustment requires Set interventions")),
    }
}

/// Fitted coefficients plus the variance model of the estimator that produced them.
struct LinearFit {
    coefficients: Vec<f64>,
    /// Raw residuals `y − Xβ̂`.
    residuals: Vec<f64>,
    rss: f64,
    variance: FitVariance,
}

/// Which sampling-variance formula describes a [`LinearFit`].
enum FitVariance {
    Ols,
    /// Bread `(XᵀX + λP)⁻¹`; no analytic SE (shrinkage bias is outside a sampling variance).
    Ridge {
        lambda: f64,
    },
    /// No analytic SE, no influence function (post-selection).
    Lasso,
    /// `scores[i] = s·ψ(r_i/s)`, `curvature[i] = ψ′(r_i/s)`.
    Huber {
        scores: Vec<f64>,
        curvature: Vec<f64>,
    },
}

/// Refuse an iterative fit that stopped at its iteration cap.
fn require_converged(converged: bool, refusal: &'static str) -> Result<(), EstimationError> {
    if converged { Ok(()) } else { Err(EstimationError::unsupported(refusal)) }
}

const LASSO_UNCONVERGED: &str =
    "Lasso coordinate descent did not converge; refuse rather than publish an unfinished fit";
const HUBER_UNCONVERGED: &str =
    "Huber M-estimator did not converge; refuse rather than publish an unfinished fit";

/// `(Σ d_i x_i x_iᵀ)⁻¹` (row-major) for nonnegative row weights `d`; `None` when singular.
fn curvature_gram_inverse(
    x_colmajor: &[f64],
    nrows: usize,
    ncols: usize,
    d: &[f64],
) -> Option<Vec<f64>> {
    let mut scaled = x_colmajor[..nrows * ncols].to_vec();
    for col in 0..ncols {
        for (v, w) in scaled[col * nrows..(col + 1) * nrows].iter_mut().zip(d) {
            *v *= w.sqrt();
        }
    }
    crate::util::xtx_inverse(&scaled, nrows, ncols)
}

/// Huber (1981, §7.10) homoskedastic-form treatment SE, used when the sandwich is
/// unavailable and the mean curvature `m = mean ψ′` is positive.
fn huber_homoskedastic_se(
    x: &[f64],
    nrows: usize,
    ncols: usize,
    t_col: usize,
    scores: &[f64],
    mean_curvature: f64,
) -> f64 {
    let n = nrows as f64;
    let p = ncols as f64;
    let kappa = 1.0 + (p / n) * (1.0 - mean_curvature) / mean_curvature;
    let tau2 =
        scores.iter().map(|e| e * e).sum::<f64>() / (n - p) / (mean_curvature * mean_curvature);
    crate::util::xtx_inverse(x, nrows, ncols)
        .map_or(f64::NAN, |inv| (kappa * kappa * tau2 * inv[t_col * ncols + t_col].max(0.0)).sqrt())
}

fn adjustment_covariates(
    data: &TabularData,
    adjustment: &[VariableId],
    expanded: &[ExpandedContrast],
    row_mask: &[bool],
) -> Result<Vec<(VariableId, Vec<f64>)>, EstimationError> {
    let mut covs = Vec::new();
    for &id in adjustment {
        if let Some(column) = expanded.iter().find(|column| column.variable == id) {
            for values in &column.columns {
                covs.push((id, mask_kept(values, row_mask)?));
            }
        } else {
            covs.push((id, data.float64_masked(id, row_mask)?));
        }
    }
    Ok(covs)
}

fn mask_kept(values: &[f64], keep: &[bool]) -> Result<Vec<f64>, EstimationError> {
    if values.len() != keep.len() {
        return Err(EstimationError::data_msg("contrast column length does not match the table"));
    }
    Ok(values.iter().zip(keep).filter_map(|(value, kept)| kept.then_some(*value)).collect())
}

/// Frisch–Waugh influence of the treatment coefficient, scaled by `delta`.
///
/// `ψ_i = δ · n · [(XᵀX)⁻¹ x_i]_T · e_i`, which equals
/// `δ · n · t̃_i e_i / Σ t̃²` with `t̃` the treatment residualized on every
/// other design column. Centering the treatment alone (`t − t̄`) is only the
/// unadjusted IF: with covariates correlated with the treatment it understates
/// the variance by `1 − R²(T | Z)`. Non-invertible designs (`xtx_inv` is
/// `None`) return NaN so a downstream mixture SE fails closed instead of
/// reporting zero.
///
/// The leverage `[(XᵀX)⁻¹ x_i]_T` accumulates one design column at a time
/// (contiguous, vectorizable) rather than striding across columns per row.
fn treatment_coef_influence(
    matrix: &[f64],
    nrows: usize,
    ncols: usize,
    t_col: usize,
    residuals: &[f64],
    delta: f64,
    xtx_inv: Option<&[f64]>,
) -> Vec<f64> {
    if nrows == 0 || residuals.len() != nrows || t_col >= ncols {
        return vec![0.0; residuals.len()];
    }
    let Some(inv) = xtx_inv else {
        return vec![f64::NAN; nrows];
    };
    let row = &inv[t_col * ncols..t_col * ncols + ncols];
    let scale = delta * nrows as f64;
    let mut psi = vec![0.0; nrows];
    for (c, &w) in row.iter().enumerate() {
        let column = &matrix[c * nrows..(c + 1) * nrows];
        for (slot, &x) in psi.iter_mut().zip(column) {
            *slot += w * x;
        }
    }
    for (slot, &e) in psi.iter_mut().zip(residuals) {
        *slot = scale * *slot * e;
    }
    psi
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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use antecedent_core::{
        AssumptionSet, AverageEffectQuery, CausalSchemaBuilder, ExecutionContext, MeasurementSpec,
        RoleHint, SmallRoleSet, TargetPopulation, ValueType, VariableId,
    };
    use antecedent_data::backend_boundary::PreparationBackend;
    use antecedent_data::materialize::{AdmissionRecord, SchemaValueKind};
    use antecedent_data::{
        Float64Column, OwnedColumn, OwnedColumnarStorage, TabularData, ValidityBitmap,
    };
    use antecedent_expr::IdentifiedEstimand;
    use antecedent_expr::{Assignment, DomainRef, EmpiricalTableProvider, ExprId, FactorSpec};
    use antecedent_identify::{DerivationTrace, IdentificationPerformanceRecord};
    use antecedent_stats::{form_xtx, invert_square};

    use crate::overlap::OverlapReport;

    use super::*;

    fn toy() -> (TabularData, IdentifiedEstimand) {
        toy_with(|_, t, z| 1.0 + 2.0 * t + z)
    }

    fn toy_with(outcome: impl Fn(usize, f64, f64) -> f64) -> (TabularData, IdentifiedEstimand) {
        let n = 100usize;
        let mut b = CausalSchemaBuilder::new();
        b.add_variable(
            "t",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::TreatmentCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        b.add_variable(
            "y",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        b.add_variable(
            "z",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::Context),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        let schema = b.build().unwrap();
        let t: Vec<f64> = (0..n).map(|i| (i % 2) as f64).collect();
        let z: Vec<f64> = (0..n).map(|i| (i as f64) / n as f64).collect();
        let y: Vec<f64> = (0..n).map(|i| outcome(i, t[i], z[i])).collect();
        let cols = vec![
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(0),
                    Arc::from(t),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(1),
                    Arc::from(y),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(2),
                    Arc::from(z),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
        ];
        let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
        let estimand = IdentifiedEstimand::backdoor(
            "backdoor.adjustment",
            Arc::from([VariableId::from_raw(2)]),
            ExprId::from_raw(0),
        );
        (TabularData::new(storage), estimand)
    }

    fn checked_result(
        query: AverageEffectQuery,
        adjustment: &[VariableId],
    ) -> IdentificationResult {
        let mut arena = CausalExprArena::new();
        let active = intervention_f64(&query.active).unwrap();
        let control = intervention_f64(&query.control).unwrap();
        let functional = arena.backdoor_ate(
            query.treatment,
            query.outcome,
            adjustment,
            antecedent_core::Value::f64(active),
            antecedent_core::Value::f64(control),
        );
        IdentificationResult::identified(
            antecedent_core::CausalQuery::average_effect(query),
            vec![IdentifiedEstimand::backdoor(
                "backdoor.adjustment",
                Arc::from(adjustment.to_vec()),
                functional,
            )],
            arena,
            DerivationTrace::default(),
            AssumptionSet::new(),
            IdentificationPerformanceRecord::default(),
        )
    }

    #[test]
    fn checked_linear_adjustment_retains_selected_target_and_assumptions() {
        let (data, _) = toy();
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let identification = checked_result(query, &[VariableId::from_raw(2)]);
        let estimator =
            LinearAdjustmentAte { bootstrap_replicates: 0, ..LinearAdjustmentAte::new() };
        let checked = estimator.prepare_checked(&data, &identification, 0).unwrap();
        assert_eq!(checked.source_functional, checked.target.functional);
        assert_ne!(checked.executable_functional, checked.source_functional);
        assert_eq!(checked.lowering.source, checked.source_functional);
        assert_eq!(checked.lowering.executable, checked.executable_functional);
        assert_eq!(checked.lowering.population, TargetPopulation::AllObserved);
        assert!(checked.factor_requirements.iter().all(|f| f.intervention.is_empty()));
        assert_eq!(
            checked.program.arena().node(checked.source_functional),
            identification.arena.node(checked.source_functional)
        );
        assert_eq!(checked.problem.adjustment_set.as_ref(), &[VariableId::from_raw(2)]);
        assert!(checked.required_assumptions.is_empty());
        let rebound = estimator.rebind_checked(&checked, &data).unwrap();
        assert_eq!(rebound.program.mapping(), checked.program.mapping());
        assert_eq!(rebound.problem.design.nrows, checked.problem.design.nrows);

        let t = [0.0, 1.0];
        let y = [0.0, 1.0];
        let z = [0.0, 1.0];
        let changed_schema = TabularData::from_f64_columns([
            ("renamed_t", t.as_slice()),
            ("y", y.as_slice()),
            ("z", z.as_slice()),
        ])
        .unwrap();
        assert!(estimator.rebind_checked(&checked, &changed_schema).is_err());
    }

    #[test]
    fn checked_observational_program_evaluates_with_arm_bindings() {
        let (data, _) = toy();
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let identification = checked_result(query, &[VariableId::from_raw(2)]);
        let estimator =
            LinearAdjustmentAte { bootstrap_replicates: 0, ..LinearAdjustmentAte::new() };
        let checked = estimator.prepare_checked(&data, &identification, 0).unwrap();

        let t = VariableId::from_raw(0);
        let y = VariableId::from_raw(1);
        let z = VariableId::from_raw(2);
        let mut provider = EmpiricalTableProvider::new();
        provider
            .set_domain(t, [antecedent_core::Value::f64(0.0), antecedent_core::Value::f64(1.0)]);
        provider
            .set_domain(y, [antecedent_core::Value::f64(0.0), antecedent_core::Value::f64(1.0)]);
        provider
            .set_domain(z, [antecedent_core::Value::f64(0.0), antecedent_core::Value::f64(1.0)]);
        for (zv, probability) in [(0.0, 0.5), (1.0, 0.5)] {
            let spec = FactorSpec {
                variables: &[z],
                conditioned_on: &[],
                intervention: &[],
                domain: DomainRef::Observational,
                population: "",
                regime: None,
            };
            provider
                .insert_probability(
                    &spec,
                    &Assignment::from_pairs([(z, antecedent_core::Value::f64(zv))]),
                    probability,
                )
                .unwrap();
        }
        for (tv, zv, py1) in [(0.0, 0.0, 0.3), (0.0, 1.0, 0.2), (1.0, 0.0, 0.8), (1.0, 1.0, 0.6)] {
            let spec = FactorSpec {
                variables: &[y],
                conditioned_on: &[t, z],
                intervention: &[],
                domain: DomainRef::Observational,
                population: "",
                regime: None,
            };
            for (yv, probability) in [(0.0, 1.0 - py1), (1.0, py1)] {
                provider
                    .insert_probability(
                        &spec,
                        &Assignment::from_pairs([
                            (t, antecedent_core::Value::f64(tv)),
                            (z, antecedent_core::Value::f64(zv)),
                            (y, antecedent_core::Value::f64(yv)),
                        ]),
                        probability,
                    )
                    .unwrap();
            }
        }
        let evaluator = checked.program.compile().unwrap();
        let context = antecedent_expr::EvalContext::default();
        let active = evaluator
            .evaluate_with(
                &provider,
                &context,
                &Assignment::from_pairs([(t, antecedent_core::Value::f64(1.0))]),
            )
            .unwrap();
        let control = evaluator
            .evaluate_with(
                &provider,
                &context,
                &Assignment::from_pairs([(t, antecedent_core::Value::f64(0.0))]),
            )
            .unwrap();
        assert!((active - 0.7).abs() < 1e-12);
        assert!((control - 0.25).abs() < 1e-12);
        assert!((active - control - 0.45).abs() < 1e-12);
    }

    #[test]
    fn checked_linear_adjustment_rejects_semantic_tampering() {
        let (data, _) = toy();
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let mut reversed = checked_result(query.clone(), &[VariableId::from_raw(2)]);
        let mut arena = reversed.arena.clone();
        let functional = arena.backdoor_ate(
            query.treatment,
            query.outcome,
            &[VariableId::from_raw(2)],
            antecedent_core::Value::f64(0.0),
            antecedent_core::Value::f64(1.0),
        );
        reversed.arena = arena;
        reversed.estimands[0].functional = functional;
        let estimator =
            LinearAdjustmentAte { bootstrap_replicates: 0, ..LinearAdjustmentAte::new() };
        assert!(estimator.prepare_checked(&data, &reversed, 0).is_err());

        let mut changed_arms = checked_result(query.clone(), &[VariableId::from_raw(2)]);
        let mut different_query = query.clone();
        different_query.active = antecedent_core::Intervention::set(
            different_query.treatment,
            antecedent_core::Value::f64(2.0),
        );
        changed_arms.query = antecedent_core::CausalQuery::average_effect(different_query);
        assert!(estimator.prepare_checked(&data, &changed_arms, 0).is_err());

        let mut changed_covariate = checked_result(query, &[VariableId::from_raw(2)]);
        changed_covariate.estimands[0].adjustment_set = Arc::from([]);
        assert!(estimator.prepare_checked(&data, &changed_covariate, 0).is_err());

        let mut changed_population =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        changed_population.target_population = TargetPopulation::Treated;
        let changed_population = checked_result(changed_population, &[VariableId::from_raw(2)]);
        assert!(estimator.prepare_checked(&data, &changed_population, 0).is_err());
    }

    #[test]
    fn overlap_report_from_propensities() {
        let ps = [0.1, 0.5, 0.9];
        let ws = [10.0, 2.0, 1.111];
        let report = OverlapReport::from_propensities(
            &ps,
            Some(&ws),
            OverlapPolicy::RequireDiagnostics { clip: Some(0.05), trim: Some(0.05) },
            Some(&[0.0, 0.0, 1.0]),
            Some(crate::overlap::IpwTarget::Ate),
            None,
        );
        assert!((report.propensity_min - 0.1).abs() < 1e-12);
        assert!((report.propensity_max - 0.9).abs() < 1e-12);
        assert_eq!(report.extreme_weight_count, 0); // none strictly > 10
        assert_eq!(report.clip, Some(0.05));
        assert!((report.excluded_fraction - 0.0).abs() < 1e-12);
        assert!((report.target_population_support - 1.0).abs() < 1e-12);
        assert_eq!(report.excluded_regions.len(), 2);
        assert!((report.excluded_regions[0].high - 0.05).abs() < 1e-12);
        let sens = report.clip_sensitivity.as_ref().expect("clip sensitivity");
        assert!(sens.thresholds.len() >= 2);
        assert_eq!(sens.ess.len(), sens.thresholds.len());
    }

    #[test]
    fn rejects_require_diagnostics_on_linear_path() {
        let (data, estimand) = toy();
        let est = LinearAdjustmentAte {
            overlap: OverlapPolicy::require_diagnostics(),
            ..LinearAdjustmentAte::new()
        };
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let err = est.prepare(&data, &estimand, &query).unwrap_err();
        assert!(matches!(err, EstimationError::Overlap { .. }));
    }

    /// The residuals the point fit hands back are the ones a separate least-squares solve of
    /// the same design produces, so the normal-equation scores built from them are identical
    /// and no second solve is needed; a shrinkage fit publishes no OLS residuals at all.
    #[test]
    fn point_fit_residuals_give_the_normal_equation_scores_without_a_second_solve() {
        let (data, estimand) = toy_with(|i, t, z| 1.0 + 2.0 * t + z + ((i * i) % 7) as f64 - 3.0);
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let est = LinearAdjustmentAte { bootstrap_replicates: 0, ..LinearAdjustmentAte::new() };
        let prep = est.prepare(&data, &estimand, &query).unwrap();
        let (point, residuals) = est
            .fit_point_with_ols_residuals(
                &prep,
                &mut EstimationWorkspace::default(),
                AssumptionSet::new(),
            )
            .unwrap();
        let residuals = residuals.expect("OLS fit carries residuals");
        let (m, n, p) = (&prep.design.matrix, prep.design.nrows, prep.design.ncols);
        let from_residuals = crate::normal_equation_scores_of_residuals(m, n, p, &residuals);
        let refit = crate::normal_equation_scores(m, n, p, &prep.design.outcome).unwrap();
        assert_eq!(from_residuals.len(), refit.len());
        for (a, b) in from_residuals.iter().flatten().zip(refit.iter().flatten()) {
            assert!((a - b).abs() < 1e-12, "{a} vs {b}");
        }
        // The residuals are non-trivial, so the comparison above is not 0 == 0.
        assert!(residuals.iter().any(|e| e.abs() > 0.5));
        assert_eq!(point.provenance.n_obs, Some(n as u64));

        let ridge = LinearAdjustmentAte {
            fit_kind: LinearFitKind::Ridge { lambda: 1.0 },
            bootstrap_replicates: 0,
            ..LinearAdjustmentAte::new()
        };
        let (_, none) = ridge
            .fit_point_with_ols_residuals(
                &prep,
                &mut EstimationWorkspace::default(),
                AssumptionSet::new(),
            )
            .unwrap();
        assert!(none.is_none());
    }

    #[test]
    fn bootstrap_replicate_failures_stay_inside_the_replicate() {
        assert!(bootstrap_fit_failed(&EstimationError::stats_msg("singular design")));
        assert!(bootstrap_fit_failed(&EstimationError::unsupported(LASSO_UNCONVERGED)));
        assert!(bootstrap_fit_failed(&EstimationError::unsupported(HUBER_UNCONVERGED)));
        assert!(bootstrap_fit_failed(&EstimationError::data_msg(
            "target population left no rows for g-computation",
        )));
        assert!(!bootstrap_fit_failed(&EstimationError::TargetPopulation));
    }

    #[test]
    fn recovers_ate_two() {
        let (data, estimand) = toy();
        let est = LinearAdjustmentAte { bootstrap_replicates: 50, ..LinearAdjustmentAte::new() };
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let prep = est.prepare(&data, &estimand, &query).unwrap();
        let mut ws = EstimationWorkspace::default();
        let ctx = ExecutionContext::for_tests(1);
        let effect = est.fit(&prep, &mut ws, &ctx, AssumptionSet::new()).unwrap();
        assert!((effect.ate - 2.0).abs() < 1e-8);
        assert!(effect.se_bootstrap().is_some());
        let identity: Vec<usize> = (0..prep.design.nrows).collect();
        let resampled = est.ate_on_row_indices(&prep, &mut ws, &identity).unwrap();
        assert!((resampled - effect.ate).abs() < 1e-12);
    }

    /// The shared Gram inverse and the column-wise leverage pass reproduce the
    /// row-by-row Frisch–Waugh influence `δ·n·[(XᵀX)⁻¹x_i]_T·e_i` and the
    /// homoskedastic SE `√(σ²·[(XᵀX)⁻¹]_TT)` exactly.
    #[test]
    fn influence_and_homoskedastic_se_match_the_row_wise_gram_formula() {
        let (data, estimand) = toy();
        let est = LinearAdjustmentAte { bootstrap_replicates: 0, ..LinearAdjustmentAte::new() };
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let prep = est.prepare(&data, &estimand, &query).unwrap();
        let mut ws = EstimationWorkspace::default();
        let ctx = ExecutionContext::for_tests(1);
        let effect = est.fit(&prep, &mut ws, &ctx, AssumptionSet::new()).unwrap();
        let (n, p) = (prep.design.nrows, prep.design.ncols);
        let t_col = prep.design.treatment_column().unwrap();
        let fit = prep.design.fit_ols(&est.backend, &mut ws.ols).unwrap();
        let mut xtx = vec![0.0; p * p];
        form_xtx(&prep.design.matrix, n, p, &mut xtx);
        let inv = invert_square(&xtx, p).unwrap();
        let sigma2 = fit.rss / (n - p) as f64;
        let se = (sigma2 * inv[t_col * p + t_col]).sqrt() * prep.treatment_delta.abs();
        assert_eq!(effect.se_analytic().to_bits(), se.to_bits());
        let influence = effect.scores.influence.as_ref().unwrap();
        let row = &inv[t_col * p..(t_col + 1) * p];
        let scale = prep.treatment_delta * n as f64;
        for i in 0..n {
            let leverage: f64 = (0..p).map(|c| row[c] * prep.design.matrix[c * n + i]).sum();
            let expected = scale * leverage * fit.residuals[i];
            assert!((influence[i] - expected).abs() == 0.0, "row {i}");
        }
    }

    #[test]
    fn scales_ate_by_level_delta() {
        let (data, estimand) = toy();
        let est = LinearAdjustmentAte { bootstrap_replicates: 0, ..LinearAdjustmentAte::new() };
        let query = AverageEffectQuery::with_levels(
            VariableId::from_raw(0),
            VariableId::from_raw(1),
            0.0,
            2.0,
        );
        let prep = est.prepare(&data, &estimand, &query).unwrap();
        let mut ws = EstimationWorkspace::default();
        let ctx = ExecutionContext::for_tests(1);
        let effect = est.fit(&prep, &mut ws, &ctx, AssumptionSet::new()).unwrap();
        // β_T ≈ 2, delta = 2 → ATE ≈ 4
        assert!((effect.ate - 4.0).abs() < 1e-8);
    }

    #[test]
    fn recovers_att_via_gcomp() {
        let (data, estimand) = toy();
        let est = LinearAdjustmentAte { bootstrap_replicates: 0, ..LinearAdjustmentAte::new() };
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1))
                .with_target_population(TargetPopulation::Treated);
        let prep = est.prepare(&data, &estimand, &query).unwrap();
        let mut ws = EstimationWorkspace::default();
        let ctx = ExecutionContext::for_tests(1);
        let effect = est.fit(&prep, &mut ws, &ctx, AssumptionSet::new()).unwrap();
        assert!((effect.ate - 2.0).abs() < 1e-8, "att={}", effect.ate);
    }

    #[test]
    fn predicate_restricts_prepared_rows() {
        use antecedent_core::PredicateExpr;
        let (data, estimand) = toy();
        let est = LinearAdjustmentAte { bootstrap_replicates: 0, ..LinearAdjustmentAte::new() };
        let all = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let half = all.clone().with_target_population(TargetPopulation::Predicate(
            PredicateExpr::rows((0..10).collect::<Vec<_>>()),
        ));
        let prep_all = est.prepare(&data, &estimand, &all).unwrap();
        let prep_half = est.prepare(&data, &estimand, &half).unwrap();
        assert_eq!(prep_all.design.nrows, 100);
        assert_eq!(prep_half.design.nrows, 10);
        let named =
            all.with_target_population(TargetPopulation::Predicate(PredicateExpr::named("cohort")));
        let err = est.prepare(&data, &estimand, &named).unwrap_err();
        assert!(
            err.to_string().contains("PopulationRegistry") || err.to_string().contains("named")
        );
    }

    #[test]
    fn source_cluster_labels_follow_their_rows_through_the_population_mask() {
        use antecedent_core::PredicateExpr;
        let (data, estimand) = toy_with(|i, t, z| 1.0 + 2.0 * t + z + wiggle(i));
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1))
                .with_target_population(TargetPopulation::Predicate(PredicateExpr::rows(
                    (10..70).collect::<Vec<_>>(),
                )));
        let labels: Vec<u32> = (0..100u32).map(|i| (i * 5) % 7).collect();
        let se_of = |est: LinearAdjustmentAte| {
            let prep = est.prepare(&data, &estimand, &query).unwrap();
            let mut ws = EstimationWorkspace::default();
            est.fit(&prep, &mut ws, &ExecutionContext::for_tests(1), AssumptionSet::new())
                .unwrap()
                .se_analytic()
        };
        let base = LinearAdjustmentAte {
            bootstrap_replicates: 0,
            se_kind: AnalyticSeKind::Cluster,
            ..LinearAdjustmentAte::new()
        };
        let by_source = se_of(base.clone().with_source_cluster_ids(labels.clone()));
        let aligned = se_of(base.clone().with_cluster_ids(labels[10..70].to_vec()));
        let misaligned = se_of(base.clone().with_cluster_ids(labels[..60].to_vec()));
        assert!(by_source.is_finite() && by_source > 0.0);
        assert_eq!(by_source.to_bits(), aligned.to_bits());
        assert_ne!(by_source.to_bits(), misaligned.to_bits());
    }

    #[test]
    fn source_cluster_labels_must_cover_the_source_rows_and_exclude_complete_case_labels() {
        let (data, estimand) = toy();
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let base = LinearAdjustmentAte {
            bootstrap_replicates: 0,
            se_kind: AnalyticSeKind::Cluster,
            ..LinearAdjustmentAte::new()
        };
        let short = base.clone().with_source_cluster_ids(vec![0; 99]);
        assert!(short.prepare(&data, &estimand, &query).unwrap_err().to_string().contains("99"));
        let both = base.with_source_cluster_ids(vec![0; 100]).with_cluster_ids(vec![0; 100]);
        assert!(
            both.prepare(&data, &estimand, &query).unwrap_err().to_string().contains("not both")
        );
    }

    #[test]
    fn hc_sandwich_kinds_yield_finite_se() {
        let (data, estimand) = toy();
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        for kind in [
            AnalyticSeKind::Hc0,
            AnalyticSeKind::Hc2,
            AnalyticSeKind::Hc3,
            AnalyticSeKind::NeweyWest { lag: 2 },
        ] {
            let est = LinearAdjustmentAte {
                bootstrap_replicates: 0,
                se_kind: kind,
                ..LinearAdjustmentAte::new()
            };
            let prep = est.prepare(&data, &estimand, &query).unwrap();
            let mut ws = EstimationWorkspace::default();
            let effect = est
                .fit(&prep, &mut ws, &ExecutionContext::for_tests(1), AssumptionSet::new())
                .unwrap();
            assert!(effect.se_analytic().is_finite() && effect.se_analytic() > 0.0, "{kind:?}");
        }
    }

    #[test]
    fn ridge_lasso_huber_fit_kinds_recover_ate() {
        let (data, estimand) = toy();
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        for kind in [
            LinearFitKind::Ridge { lambda: 1e-3 },
            LinearFitKind::Lasso { lambda: 1e-4 },
            LinearFitKind::Huber { c: 1.345 },
        ] {
            let est = LinearAdjustmentAte {
                bootstrap_replicates: 0,
                fit_kind: kind,
                ..LinearAdjustmentAte::new()
            };
            let prep = est.prepare(&data, &estimand, &query).unwrap();
            let mut ws = EstimationWorkspace::default();
            let effect = est
                .fit(&prep, &mut ws, &ExecutionContext::for_tests(2), AssumptionSet::new())
                .unwrap();
            assert!(effect.ate.is_finite(), "{kind:?}");
            assert!((effect.ate - 2.0).abs() < 0.05, "ate={} kind={kind:?}", effect.ate);
        }
    }

    #[test]
    fn lasso_analytic_se_nan_bootstrap_finite() {
        let (data, estimand) = toy();
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let no_boot = LinearAdjustmentAte {
            bootstrap_replicates: 0,
            fit_kind: LinearFitKind::Lasso { lambda: 1e-4 },
            ..LinearAdjustmentAte::new()
        };
        let prep = no_boot.prepare(&data, &estimand, &query).unwrap();
        let mut ws = EstimationWorkspace::default();
        let effect = no_boot
            .fit(&prep, &mut ws, &ExecutionContext::for_tests(3), AssumptionSet::new())
            .unwrap();
        assert!(effect.se_analytic().is_nan(), "lasso se_analytic={}", effect.se_analytic());
        assert!(effect.se_bootstrap().is_none());

        let with_boot = LinearAdjustmentAte {
            bootstrap_replicates: 40,
            fit_kind: LinearFitKind::Lasso { lambda: 1e-4 },
            ..LinearAdjustmentAte::new()
        };
        let effect = with_boot
            .fit(&prep, &mut ws, &ExecutionContext::for_tests(3), AssumptionSet::new())
            .unwrap();
        assert!(effect.se_analytic().is_nan());
        let boot = effect.se_bootstrap().expect("bootstrap SE");
        assert!(boot.is_finite() && boot >= 0.0, "se_bootstrap={boot}");
    }

    /// Deterministic bounded noise in `[-0.1, 0.1]`.
    fn wiggle(i: usize) -> f64 {
        0.1 * (7.31 * i as f64).sin()
    }

    fn fit_kind_effect(
        data: &TabularData,
        estimand: &IdentifiedEstimand,
        fit_kind: LinearFitKind,
        se_kind: AnalyticSeKind,
    ) -> EffectEstimate {
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let est = LinearAdjustmentAte {
            bootstrap_replicates: 0,
            fit_kind,
            se_kind,
            ..LinearAdjustmentAte::new()
        };
        let prep = est.prepare(data, estimand, &query).unwrap();
        let mut ws = EstimationWorkspace::default();
        est.fit(&prep, &mut ws, &ExecutionContext::for_tests(5), AssumptionSet::new()).unwrap()
    }

    #[test]
    fn huber_with_unclipped_scores_reproduces_ols_variance_and_influence() {
        // With c so large that no residual is clipped, ψ(u)=u and ψ′=1, so the Huber
        // sandwich collapses to the OLS one: τ² = RSS/(n−p), κ = 1, bread (XᵀX)⁻¹.
        let (data, estimand) = toy_with(|i, t, z| 1.0 + 2.0 * t + z + wiggle(i));
        for se_kind in [AnalyticSeKind::Homoskedastic, AnalyticSeKind::Hc0, AnalyticSeKind::Hc3] {
            let ols = fit_kind_effect(&data, &estimand, LinearFitKind::Ols, se_kind);
            let hub = fit_kind_effect(&data, &estimand, LinearFitKind::Huber { c: 1e9 }, se_kind);
            assert!((hub.ate - ols.ate).abs() < 1e-8, "{se_kind:?}");
            let rel = (hub.se_analytic() - ols.se_analytic()).abs() / ols.se_analytic();
            assert!(
                rel < 1e-7,
                "{se_kind:?}: huber {} ols {}",
                hub.se_analytic(),
                ols.se_analytic()
            );
            let (hi, oi) =
                (hub.scores.influence.as_ref().unwrap(), ols.scores.influence.as_ref().unwrap());
            for (a, b) in hi.iter().zip(oi.iter()) {
                assert!((a - b).abs() < 1e-6 * (1.0 + b.abs()));
            }
        }
    }

    #[test]
    fn huber_se_ignores_gross_outliers_the_estimator_downweights() {
        // 10% of rows (one odd, one even index per 20, so both arms) carry a +50 shock. OLS
        // RSS/(n−p) is dominated by them; Huber's τ² = E[ψ²]/E[ψ′]² only sees them clipped
        // at c·s, with s ≈ MAD-scale of the ±0.1 noise, so its SE must be far below the OLS
        // one and the effect stays near 2.
        let (data, estimand) = toy_with(|i, t, z| {
            let shock = if i % 20 == 3 || i % 20 == 8 { 50.0 } else { 0.0 };
            1.0 + 2.0 * t + z + wiggle(i) + shock
        });
        let ols =
            fit_kind_effect(&data, &estimand, LinearFitKind::Ols, AnalyticSeKind::Homoskedastic);
        let hub = fit_kind_effect(
            &data,
            &estimand,
            LinearFitKind::Huber { c: 1.345 },
            AnalyticSeKind::Homoskedastic,
        );
        assert!((hub.ate - 2.0).abs() < 0.1, "huber ate {}", hub.ate);
        assert!(
            hub.se_analytic() < 0.1 * ols.se_analytic(),
            "huber se {} vs ols se {}",
            hub.se_analytic(),
            ols.se_analytic()
        );
    }

    #[test]
    fn ridge_and_lasso_publish_no_analytic_se_and_no_ols_influence() {
        let (data, estimand) = toy_with(|i, t, z| 1.0 + 2.0 * t + z + wiggle(i));
        let ols =
            fit_kind_effect(&data, &estimand, LinearFitKind::Ols, AnalyticSeKind::Homoskedastic);
        let ridge0 = fit_kind_effect(
            &data,
            &estimand,
            LinearFitKind::Ridge { lambda: 0.0 },
            AnalyticSeKind::Homoskedastic,
        );
        assert!(ridge0.se_analytic().is_nan());
        // λ = 0 ridge is OLS, so its influence must be the OLS one.
        let (ri, oi) =
            (ridge0.scores.influence.as_ref().unwrap(), ols.scores.influence.as_ref().unwrap());
        for (a, b) in ri.iter().zip(oi.iter()) {
            assert!((a - b).abs() < 1e-6 * (1.0 + b.abs()));
        }
        // Heavy ridge shrinks the treatment coefficient, so its bread differs from (XᵀX)⁻¹.
        let ridge = fit_kind_effect(
            &data,
            &estimand,
            LinearFitKind::Ridge { lambda: 20.0 },
            AnalyticSeKind::Homoskedastic,
        );
        assert!(ridge.se_analytic().is_nan());
        assert!(ridge.ate < ols.ate - 1e-3);
        let heavy = ridge.scores.influence.as_ref().unwrap();
        assert!(heavy.iter().zip(oi.iter()).any(|(a, b)| (a - b).abs() > 1e-3));
        let lasso = fit_kind_effect(
            &data,
            &estimand,
            LinearFitKind::Lasso { lambda: 1e-4 },
            AnalyticSeKind::Homoskedastic,
        );
        assert!(lasso.se_analytic().is_nan());
        assert!(lasso.scores.influence.is_none());
    }

    #[test]
    fn unconverged_iterative_fits_are_refused() {
        assert!(require_converged(true, LASSO_UNCONVERGED).is_ok());
        for msg in [LASSO_UNCONVERGED, HUBER_UNCONVERGED] {
            let err = require_converged(false, msg).unwrap_err();
            assert!(err.to_string().contains("did not converge"), "{err}");
            assert!(bootstrap_fit_failed(&err));
        }
    }

    #[test]
    fn bootstrap_skips_singular_fits_and_empty_arms_only() {
        assert!(bootstrap_fit_failed(&EstimationError::stats_msg("singular design")));
        assert!(bootstrap_fit_failed(&EstimationError::data_msg(
            "target population left no rows for g-computation",
        )));
        assert!(!bootstrap_fit_failed(&EstimationError::data_msg(
            "row resample buffers must match the row map",
        )));
    }

    #[test]
    fn the_estimate_records_the_preparation_it_was_fitted_on() {
        let (data, estimand) = toy();
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let est = LinearAdjustmentAte::new();
        let prepared = est.prepare(&data, &estimand, &query).unwrap();
        let point = est
            .fit_point(&prepared, &mut EstimationWorkspace::default(), AssumptionSet::new())
            .unwrap();
        crate::prepare::assert_records_preparation(&point, 100);
    }

    #[test]
    fn prepare_records_order_schema_missing_values_and_a_proved_bound() {
        let (data, estimand) = toy();
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let prepared = LinearAdjustmentAte::new().prepare(&data, &estimand, &query).unwrap();
        let receipt = &prepared.preparation;
        assert_eq!(receipt.path(), PreparationBackend::DirectArrow);
        assert_eq!(
            receipt.schema().columns.iter().map(|column| column.name.as_ref()).collect::<Vec<_>>(),
            ["t", "y", "z"]
        );
        assert_eq!(receipt.schema().columns[0].value_kind, SchemaValueKind::Continuous);
        assert!(receipt.categories().is_empty());
        assert_eq!(receipt.missing().unwrap().proved_rows_kept(), Some(data.row_count()));
        assert_eq!(receipt.proved_rows(), Some(u64::try_from(data.row_count()).unwrap()));
        assert!(!receipt.estimator_sees_unbounded_stream());
        // No budget was declared, so the payload is unmetered and the route is not bounded.
        assert!(matches!(receipt.admission(), AdmissionRecord::Unmetered { .. }));
        assert!(!receipt.is_bounded_end_to_end_route());

        let budgeted = LinearAdjustmentAte::new()
            .with_preparation_budget(PreparationBudget::new(1 << 20))
            .prepare(&data, &estimand, &query)
            .unwrap();
        assert!(budgeted.preparation.is_bounded_end_to_end_route());
        assert_eq!(budgeted.preparation.proved_rows(), receipt.proved_rows());
    }

    #[test]
    fn a_tight_budget_refuses_preparation() {
        let (data, estimand) = toy();
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let estimator =
            LinearAdjustmentAte::new().with_preparation_budget(PreparationBudget::new(0));
        let message = estimator.prepare(&data, &estimand, &query).expect_err("budget").to_string();
        assert!(message.contains("exceeds budget 0"), "{message}");
    }

    #[test]
    fn linear_adjustment_refuses_a_scan_that_is_not_a_bounded_route() {
        use antecedent_data::materialize::{
            EstimatorStreamNeed, PreparationBudget, StreamDelivery, StreamQuote, deliver_stream,
        };

        let quote = StreamQuote {
            scratch_bytes: Some(8),
            materialize_bytes: Some(8),
            chunk_rows: vec![4, 4],
        };
        let mut controls = PreparationControls::finite_resident();
        controls.stream = EstimatorStreamNeed::CanStream;
        controls.budget = Some(PreparationBudget::new(8));
        let StreamDelivery::Scan(scan) =
            deliver_stream(&quote, PreparationBudget::new(8), controls).unwrap()
        else {
            panic!("scan");
        };
        let refused = LinearAdjustmentAte::reject_stream(&scan);
        assert!(refused.to_string().contains("cannot stream"), "{refused}");
        assert!(scan.receipt().estimator_sees_unbounded_stream());
        assert!(!scan.receipt().is_bounded_end_to_end_route());
    }

    fn categorical_adjustment_table() -> (TabularData, IdentifiedEstimand) {
        use antecedent_core::CategoryDomainId;
        use antecedent_data::{
            CategoricalColumn, CategoryCode, CategoryDomain, CategoryLevel, UnknownCategoryPolicy,
            ValidityBitmap,
        };

        let domain = CategoryDomain::try_new(
            CategoryDomainId::from_raw(2),
            vec![CategoryLevel { label: Arc::from("a") }, CategoryLevel { label: Arc::from("b") }],
            false,
            Some(CategoryCode::from_raw(0)),
            UnknownCategoryPolicy::Fail,
        )
        .unwrap();
        let n = 4usize;
        let mut builder = CausalSchemaBuilder::new();
        builder
            .add_variable(
                "t",
                ValueType::Continuous,
                SmallRoleSet::from_hint(RoleHint::TreatmentCandidate),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
        builder
            .add_variable(
                "y",
                ValueType::Continuous,
                SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
        builder
            .add_variable(
                "z",
                ValueType::Categorical,
                SmallRoleSet::from_hint(RoleHint::Context),
                None,
                Some(CategoryDomainId::from_raw(2)),
                MeasurementSpec::default(),
            )
            .unwrap();
        let schema = builder.build().unwrap();
        let columns = vec![
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(0),
                    Arc::<[f64]>::from(vec![0.0, 1.0, 0.0, 1.0]),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(1),
                    Arc::<[f64]>::from(vec![0.0, 1.0, 1.0, 2.0]),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
            OwnedColumn::Categorical(
                CategoricalColumn::try_new(
                    VariableId::from_raw(2),
                    vec![
                        CategoryCode::from_raw(0),
                        CategoryCode::from_raw(0),
                        CategoryCode::from_raw(1),
                        CategoryCode::from_raw(1),
                    ],
                    ValidityBitmap::all_valid(n),
                    Arc::new(domain),
                )
                .unwrap(),
            ),
        ];
        let data =
            TabularData::new(OwnedColumnarStorage::try_new(schema, columns, None, None).unwrap());
        let estimand = IdentifiedEstimand::backdoor(
            "backdoor.adjustment",
            Arc::from([VariableId::from_raw(2)]),
            ExprId::from_raw(0),
        );
        (data, estimand)
    }

    #[test]
    fn a_categorical_adjustment_is_expanded_before_the_design() {
        use antecedent_data::{CategoryCode, Contrast};
        let (data, estimand) = categorical_adjustment_table();
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let estimator = LinearAdjustmentAte::new().with_contrast(
            VariableId::from_raw(2),
            Contrast::Treatment { reference: CategoryCode::from_raw(0) },
        );
        let prepared = estimator.prepare(&data, &estimand, &query).unwrap();
        assert_eq!(prepared.design.ncols, 3);
        assert_eq!(prepared.preparation.categories().len(), 1);
        assert_eq!(prepared.preparation.categories()[0].coding, "treatment");
        assert_eq!(prepared.preparation.categories()[0].n_columns, 1);
    }

    #[test]
    fn a_repeated_categorical_is_expanded_and_recorded_once() {
        use antecedent_data::{CategoryCode, Contrast};
        let (data, _) = categorical_adjustment_table();
        let contrast = Contrast::Treatment { reference: CategoryCode::from_raw(0) };
        let ids = [
            VariableId::from_raw(0),
            VariableId::from_raw(1),
            VariableId::from_raw(2),
            VariableId::from_raw(2),
        ];
        let product = prepare_table(
            &data,
            &ids,
            MissingPolicy::CompleteCase,
            &[(VariableId::from_raw(2), contrast)],
            PreparationControls::finite_resident(),
        )
        .unwrap();
        assert_eq!(product.expanded.len(), 1);
        assert_eq!(product.receipt.categories().len(), 1);
        assert_eq!(product.receipt.schema().columns.len(), 3);
        assert_eq!(product.receipt.category_levels().len(), 1);
        assert_eq!(product.row_mask, vec![true; data.row_count()]);
    }
}
