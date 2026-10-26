//! Augmented inverse-probability weighting (AIPW / doubly robust) ATE estimator.
//!
//! Combines an outcome regression (μ0(Z), μ1(Z), fit separately per treatment arm) with
//! inverse-probability weighting of the residuals, so the estimator is consistent if *either*
//! the propensity model or the outcome model is correctly specified:
//!
//! ```text
//! ψ_i = (μ1(Z_i) − μ0(Z_i)) + T_i/e_i · (Y_i − μ1(Z_i)) − (1−T_i)/(1−e_i) · (Y_i − μ0(Z_i))
//! ATE = mean(ψ)
//! ```
//!
//! When the overlap policy sets a trim threshold, units whose raw propensity falls outside
//! `[trim, 1 − trim]` are excluded from the outcome-model fits and the ψ average (the estimand
//! becomes the common-support population, matching the overlap report).
//!
//! Positivity is mandatory — [`OverlapPolicy::ExplicitOverride`] is refused, matching the other
//! propensity-based estimators in [`crate::propensity`].
//!
//! Untrimmed `AllObserved` fits use the same cross-fitted score table as `retarget` for
//! every SE kind, so the point estimate does not depend on which SE was requested; the
//! table's covariance and simultaneous bands are exported only under the iid SE. ATT/ATC,
//! trimmed, and predicate-target fits stay on the full-sample path and do not export that table.
//!
//! An explicit penalized propensity ([`PropensityNuisance`]) is a separate, narrower route:
//! the untrimmed `AllObserved` mean ATE only, with a ridge- or lasso-logistic penalty (and, for
//! the lasso, the selected support) chosen on each cross-fit fold's training rows. It keeps the
//! score table, row identity, overlap report and retargetability, and publishes the
//! cross-fitted influence-function SE of the out-of-fold scores or, with
//! `bootstrap_replicates > 0`, a bootstrap SE that repeats penalty selection and every nuisance
//! fit on each resample. `docs/guides/penalized-aipw.md` states the remainder condition each
//! rests on and what is not shown. A declared GLM-to-penalized fallback re-runs the whole
//! route with the penalized destination when the GLM propensity fit fails.
//!
//! Analytic SEs on the full-sample path correct ψ for parametric nuisances:
//! ATE-type targets add the exact stacked-M terms for the logistic and the two arm OLS fits
//! (valid under a misspecified propensity too); ATT/ATC use the
//! centered efficient influence function `(N_i − τ·T_i)/π`. That function
//! accounts for the estimated propensity and arm share only when *both*
//! nuisance models are consistent: the ATT/ATC point estimate stays doubly
//! robust, but its analytic SE does not, because under a misspecified
//! propensity or outcome model the estimator's influence function picks up
//! nuisance-estimation terms the efficient influence function omits.
//! The analytic SEs are also not valid for flexible / nonparametric nuisances; default
//! inference is the 200-replicate bootstrap, which refits ê, μ̂₀, and μ̂₁ on
//! every resample.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![cfg_attr(
    test,
    allow(
        clippy::cast_possible_truncation,
        reason = "test fixtures compare exact constants and index with small literals"
    )
)]

use antecedent_core::{
    AssumptionSet, AverageEffectQuery, ExecutionContext, PopulationRegistry, TargetPopulation,
};
use antecedent_data::{TableView, TabularData};
use antecedent_expr::IdentifiedEstimand;
use antecedent_expr::{
    FactorRequirement, FunctionalProgram, ProgramLimits, ProgramSchema, ProgramVariable,
};
use antecedent_identify::{IdentificationResult, IdentificationStatus};
use antecedent_stats::{
    DenseLinearAlgebra, FaerBackend, GlmOptions, LeastSquaresWorkspace, PropensityWorkspace,
};
use std::borrow::Cow;
use std::sync::Arc;

use crate::adjustment::EffectEstimate;
use crate::bootstrap_wave::retained_index_slots;
use crate::cluster_dml_aipw::ClusterDml;
use crate::error::EstimationError;
use crate::overlap::{IpwTarget, OverlapPolicy, OverlapReport};
use crate::propensity::{
    OtherLevels, PenalizedReport, PenalizedVarianceReport, PreparedPropensityProblem,
    PropensityModel, PropensityNuisance, REFIT_BOOTSTRAP_UNCERTAINTY_KIND, ReplicatePenalties,
    clamp_scores, clip_of, crossfit_influence_se, default_propensity_overlap, fold_supports,
    gather, gather_into, prepare_propensity_problem_with_levels, split_by_treatment, trim_of,
    trim_retained_rows,
};
use crate::sampling_weights::SamplingWeights;
use crate::se::AnalyticSeKind;
use crate::util::{BootstrapOutcome, refuse};

/// Reusable scratch for AIPW point-estimate and bootstrap fits.
///
/// Outcome regressions are refit per treatment arm on every call; the design/outcome buffers
/// below are reused (grow-only) across bootstrap replicates to avoid per-replicate heap churn.
#[derive(Clone, Debug, Default)]
pub struct AipwWorkspace {
    /// Logistic IRLS scratch reused across propensity refits.
    pub propensity: PropensityWorkspace,
    /// OLS scratch reused across both arms' outcome-model refits.
    pub outcome: LeastSquaresWorkspace,
    treated_design: Vec<f64>,
    treated_outcome: Vec<f64>,
    control_design: Vec<f64>,
    control_outcome: Vec<f64>,
    pub(crate) mu0: Vec<f64>,
    pub(crate) mu1: Vec<f64>,
    psi: Vec<f64>,
}

/// Licensed procedure retained by the checked frequentist AIPW ATE route.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CheckedAipwProcedure {
    /// Logistic propensity and arm-specific OLS outcome nuisances with cross-fitted scores.
    CrossFittedLogisticOls,
    /// Logistic propensity and arm-specific OLS outcome nuisances refit on the
    /// common-support rows a trim rule retains; full-sample residualized, no score table.
    TrimmedLogisticOls,
}

impl CheckedAipwProcedure {
    /// The procedure an overlap policy selects: the trim rule decides between the
    /// cross-fitted score table and the full-sample common-support refit.
    #[must_use]
    pub fn for_overlap(overlap: OverlapPolicy) -> Self {
        if trim_of(overlap).is_some() {
            Self::TrimmedLogisticOls
        } else {
            Self::CrossFittedLogisticOls
        }
    }

    /// Cross-fit fold count the procedure runs; zero for the full-sample refit.
    #[must_use]
    pub const fn folds(self) -> usize {
        match self {
            Self::CrossFittedLogisticOls => crate::crossfit_aipw::DEFAULT_AIPW_FOLDS,
            Self::TrimmedLogisticOls => 0,
        }
    }

    /// Stable wire spelling of the procedure.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::CrossFittedLogisticOls => "cross_fitted_logistic_ols",
            Self::TrimmedLogisticOls => "trimmed_logistic_ols",
        }
    }
}

/// Selected target roles and numerical choices for checked AIPW execution.
#[derive(Clone, Debug, PartialEq)]
pub struct CheckedAipwLowering {
    /// Source causal target root.
    pub functional: antecedent_expr::ExprId,
    /// Treatment role.
    pub treatment: antecedent_core::VariableId,
    /// Outcome role.
    pub outcome: antecedent_core::VariableId,
    /// Adjustment roles in selected-estimand order.
    pub adjustment: Arc<[antecedent_core::VariableId]>,
    /// Population whose mean contrast is estimated.
    pub population: TargetPopulation,
    /// Estimator procedure retained for execution.
    pub procedure: CheckedAipwProcedure,
    /// Overlap policy the procedure clips and trims with; its trim rule selects
    /// `procedure`.
    pub overlap: OverlapPolicy,
    /// Cross-fit fold count (zero for the trimmed full-sample procedure).
    pub folds: usize,
    /// Analytic uncertainty method.
    pub se_kind: AnalyticSeKind,
    /// Bootstrap replicate count (zero disables bootstrap uncertainty).
    pub bootstrap_replicates: u32,
    /// Declared binary-propensity nuisance. A penalized one keeps the cross-fitted
    /// `procedure` but withholds every interval; the configuration is part of the receipt.
    pub propensity: PropensityNuisance,
    /// Declared cluster-DML independence unit, when the cross-fit owns whole clusters.
    pub cluster_dml: Option<ClusterDml>,
    /// Complete-case source row identities bound during preparation.
    pub rows: Arc<[antecedent_data::GlobalId]>,
}

/// Checked AIPW preparation sealed to one identified ATE claim and data binding.
#[derive(Clone, Debug)]
pub struct CheckedAipwPreparation {
    target: IdentifiedEstimand,
    program: FunctionalProgram,
    factor_requirements: Arc<[FactorRequirement]>,
    lowering: CheckedAipwLowering,
    query: AverageEffectQuery,
    required_assumptions: AssumptionSet,
    problem: PreparedPropensityProblem,
}

impl CheckedAipwPreparation {
    /// Selected identified target.
    #[must_use]
    pub fn target(&self) -> &IdentifiedEstimand {
        &self.target
    }
    /// Checked owner of the selected causal functional.
    #[must_use]
    pub fn program(&self) -> &FunctionalProgram {
        &self.program
    }
    /// Checked provider requirements of the retained target program.
    #[must_use]
    pub fn factor_requirements(&self) -> &[FactorRequirement] {
        &self.factor_requirements
    }
    /// Retained typed procedure and role bindings.
    #[must_use]
    pub fn lowering(&self) -> &CheckedAipwLowering {
        &self.lowering
    }
    /// Selected identification assumptions.
    #[must_use]
    pub fn required_assumptions(&self) -> &AssumptionSet {
        &self.required_assumptions
    }
    /// Bound numerical design, including complete-case row identities.
    #[must_use]
    pub fn problem(&self) -> &PreparedPropensityProblem {
        &self.problem
    }
}

/// Doubly robust (AIPW) ATE / ATT / ATC estimator.
///
/// Supports [`TargetPopulation::AllObserved`], [`TargetPopulation::Treated`] (ATT), and
/// [`TargetPopulation::Untreated`] (ATC). Positivity is mandatory:
/// [`OverlapPolicy::ExplicitOverride`] is refused.
#[derive(Clone, Debug)]
pub struct AipwAte {
    /// Dense linear-algebra backend used for the propensity IRLS fit and outcome OLS fits.
    pub backend: FaerBackend,
    /// Bootstrap replicates (0 = skip bootstrap).
    pub bootstrap_replicates: u32,
    /// Overlap policy; must be [`OverlapPolicy::RequireDiagnostics`].
    pub overlap: OverlapPolicy,
    /// GLM fitting options for the propensity model.
    pub glm_options: GlmOptions,
    /// Analytic SE kind (IID influence / HC1 scale / cluster).
    pub se_kind: AnalyticSeKind,
    /// Optional cluster ids for [`AnalyticSeKind::Cluster`] (aligned to prepared rows).
    pub cluster_ids: Option<Vec<u32>>,
    /// Optional cluster labels of every row of the source table for
    /// [`AnalyticSeKind::Cluster`], subset to the complete-case rows by their original
    /// index. Exclusive with [`Self::cluster_ids`]. With a [`ClusterDml`] declaration they are
    /// the first-endpoint unit labels, mapped to the complete-case rows before the cross-fit
    /// plan is built.
    pub source_cluster_ids: Option<Vec<u32>>,
    /// Sampling design of the rows, for the bootstrap. Undeclared, the design is the one the
    /// cluster labels or panel times imply; declared, a resample the design does not allow
    /// (a row-level draw under a trajectory design, say) is refused.
    pub sampling_design: crate::util::SamplingDeclaration,
    /// Second-endpoint labels of a dyadic [`ClusterDml`] declaration for every row of the
    /// source table. Exclusive with [`Self::cluster_ids_second`].
    pub source_cluster_ids_second: Option<Vec<u32>>,
    /// Second-endpoint labels of a dyadic [`ClusterDml`] declaration (aligned to prepared
    /// rows); [`Self::cluster_ids`] are then the first endpoints.
    pub cluster_ids_second: Option<Vec<u32>>,
    /// Optional bindings for named predicates / custom target distributions.
    pub population_registry: Option<PopulationRegistry>,
    /// Multiway cluster ids (one `Vec<u32>` per clustering dimension).
    pub multiway_ids: Option<Vec<Vec<u32>>>,
    /// Optional panel time labels for [`AnalyticSeKind::PanelClusterHac`].
    pub panel_times: Option<Vec<i64>>,
    /// Declared binary-propensity nuisance: the unpenalized logistic by default, or an
    /// explicit ridge-logistic penalty (distinct from `glm_options.ridge_on_separation`).
    pub propensity: PropensityNuisance,
    /// Declared cluster-DML independence unit over [`Self::cluster_ids`]: whole clusters are
    /// cross-fit together and no interval is published (see [`ClusterDml`]).
    pub cluster_dml: Option<ClusterDml>,
    /// Policy for treatment rows at neither declared level. The default refuses; excluding
    /// them restricts the target population to the rows at the two declared levels (the
    /// pairwise-subpopulation target) and must be declared. The checked route refuses it.
    pub other_levels: OtherLevels,
    /// Sampling weights of every row of the source table (see [`crate::sampling_weights`]).
    /// The cross-fitted ATE route reads them: nuisances are fit by weighted logistic and
    /// weighted least squares, the scores are averaged under the weights, and the standard
    /// error is the linearization of that weighted mean. Trimmed, ATT/ATC, penalized and
    /// cluster-DML constructions refuse them.
    pub sampling_weights: Option<SamplingWeights>,
    /// A requested bootstrap interval. When set, the non-penalized bootstrap runs as the wave
    /// procedure that stops on the request's Monte Carlo tolerances (capped by
    /// [`Self::bootstrap_replicates`]) and reports its interval, waves, Monte Carlo error and
    /// stop reason in [`crate::EstimateDiagnostics::resampling`]. The penalized refit
    /// bootstrap, which records per-replicate penalties, keeps its fixed count. `None` runs
    /// the fixed-count bootstrap exactly as before.
    pub bootstrap_interval: Option<crate::util::IntervalRequest>,
}

/// Per-row influence of the control-to-treated score contrast: the plain score difference
/// under uniform weights, otherwise the target-weighted centered difference.
fn contrast_influence(
    table: &crate::scores::ScoreTable,
    weights: Option<&[f64]>,
    contrast_value: f64,
) -> Result<Vec<f64>, EstimationError> {
    let n = table.n_rows as f64;
    Ok(match weights {
        Some(w) => {
            let mass: f64 = w.iter().sum();
            table
                .column(0)?
                .iter()
                .zip(table.column(1)?)
                .zip(w)
                .map(|((&a, &b), &wi)| n * wi / mass * (b - a - contrast_value))
                .collect()
        }
        None => table.column(0)?.iter().zip(table.column(1)?).map(|(&a, &b)| b - a).collect(),
    })
}

impl Default for AipwAte {
    fn default() -> Self {
        Self::new()
    }
}

impl AipwAte {
    /// Defaults: 200 bootstrap replicates, clip = 0.01, no trim.
    #[must_use]
    pub fn new() -> Self {
        Self {
            backend: FaerBackend,
            bootstrap_replicates: 200,
            overlap: default_propensity_overlap(),
            glm_options: GlmOptions::default(),
            se_kind: AnalyticSeKind::Homoskedastic,
            cluster_ids: None,
            source_cluster_ids: None,
            sampling_design: crate::util::SamplingDeclaration::Undeclared,
            source_cluster_ids_second: None,
            cluster_ids_second: None,
            population_registry: None,
            multiway_ids: None,
            panel_times: None,
            propensity: PropensityNuisance::default(),
            cluster_dml: None,
            other_levels: OtherLevels::default(),
            sampling_weights: None,
            bootstrap_interval: None,
        }
    }

    crate::inference::with_bootstrap_interval!();

    /// Declare sampling weights for every row of the source table. They are read by the
    /// cross-fitted ATE route; the estimand is the effect in the weighted population.
    #[must_use]
    pub fn with_sampling_weights(mut self, weights: SamplingWeights) -> Self {
        self.sampling_weights = Some(weights);
        self
    }

    /// Declare what to do with rows whose treatment is at neither declared level (see
    /// [`OtherLevels`]). The default refuses.
    #[must_use]
    pub const fn with_other_levels(mut self, other_levels: OtherLevels) -> Self {
        self.other_levels = other_levels;
        self
    }

    /// Declare the binary-propensity nuisance (see [`PropensityNuisance`]).
    ///
    /// A ridge- or lasso-logistic penalty is licensed for the untrimmed `AllObserved` mean ATE:
    /// it publishes the cross-fitted point estimate, the score table and the intervals
    /// described in the module docs. A ridge or lasso fallback re-runs that route when the GLM
    /// fit fails; an ML fallback is declarable but closed.
    #[must_use]
    pub fn with_propensity_nuisance(mut self, propensity: PropensityNuisance) -> Self {
        self.propensity = propensity;
        self
    }

    /// Declare the independence unit of the cross-fit: whole clusters of [`Self::cluster_ids`]
    /// (which this does not set) share a fold, and the route publishes the cross-fitted point
    /// estimate and score table with no interval. Licensed for the untrimmed `AllObserved`
    /// mean ATE with `bootstrap_replicates == 0`; see [`ClusterDml`].
    #[must_use]
    pub const fn with_cluster_dml(mut self, cluster_dml: ClusterDml) -> Self {
        self.cluster_dml = Some(cluster_dml);
        self
    }

    crate::inference::with_bootstrap_replicates!();

    /// Set the overlap policy. Positivity is mandatory here:
    /// [`OverlapPolicy::ExplicitOverride`] is refused by `prepare`.
    #[must_use]
    pub const fn with_overlap(mut self, overlap: OverlapPolicy) -> Self {
        self.overlap = overlap;
        self
    }

    /// Set the GLM fitting options (max iterations, convergence tolerance) used for the
    /// propensity model.
    #[must_use]
    pub const fn with_glm_options(mut self, glm_options: GlmOptions) -> Self {
        self.glm_options = glm_options;
        self
    }

    crate::inference::with_se_kind!();

    crate::inference::with_cluster_ids!();

    crate::inference::with_source_cluster_ids!();

    crate::inference::with_sampling_design!();

    /// Set the second-endpoint labels of a dyadic [`ClusterDml`] declaration for every row of
    /// the source table (length must equal its row count).
    #[must_use]
    pub fn with_source_second_cluster_ids(mut self, cluster_ids: Vec<u32>) -> Self {
        self.source_cluster_ids_second = Some(cluster_ids);
        self
    }

    /// The `(first, second)` unit labels of the complete-case rows `row_index`, from labels
    /// declared per retained row or per source row.
    ///
    /// # Errors
    ///
    /// A retained row without a declared source label.
    pub fn unit_labels(
        &self,
        row_index: &[antecedent_data::GlobalId],
    ) -> Result<crate::cluster_dml_aipw::UnitLabels, EstimationError> {
        crate::cluster_dml_aipw::declared_unit_labels(
            (self.cluster_ids.as_deref(), self.source_cluster_ids.as_deref()),
            (self.cluster_ids_second.as_deref(), self.source_cluster_ids_second.as_deref()),
            row_index,
        )
    }

    /// Whether the declared units are dyadic (two endpoint label sets).
    #[must_use]
    pub fn has_second_units(&self) -> bool {
        self.cluster_ids_second.is_some() || self.source_cluster_ids_second.is_some()
    }

    /// Set the second-endpoint labels of a dyadic [`ClusterDml`] declaration.
    #[must_use]
    pub fn with_second_cluster_ids(mut self, cluster_ids: Vec<u32>) -> Self {
        self.cluster_ids_second = Some(cluster_ids);
        self
    }

    crate::inference::with_multiway_ids!();

    crate::inference::with_panel_times!();

    /// Prepare the covariate design from tabular data, identified estimand, and query.
    ///
    /// Accepts `backdoor.adjustment` / `backdoor.efficient` estimands.
    ///
    /// # Errors
    ///
    /// Overlap policy is `ExplicitOverride`, incompatible estimand, unsupported query, or
    /// missing/invalid data columns.
    pub fn prepare(
        &self,
        data: &TabularData,
        estimand: &IdentifiedEstimand,
        query: &AverageEffectQuery,
    ) -> Result<PreparedPropensityProblem, EstimationError> {
        self.propensity.validate_for_execution()?;
        if self.propensity.is_penalized() {
            self.require_penalized_scope(
                query.outcome_functional == antecedent_core::OutcomeFunctional::Mean,
                &query.target_population,
                self.overlap,
            )?;
        }
        if self.sampling_weights.is_some() {
            Self::require_weighted_scope(query)?;
        }
        if let Some(source) = self.source_cluster_ids.as_deref() {
            crate::se::require_source_labels(
                source,
                self.cluster_ids.as_deref(),
                data.row_count(),
            )?;
        }
        if let Some(source) = self.source_cluster_ids_second.as_deref() {
            crate::se::require_source_labels(
                source,
                self.cluster_ids_second.as_deref(),
                data.row_count(),
            )?;
        }
        if self.cluster_dml.is_some() {
            crate::cluster_dml_aipw::require_unpenalized(self.propensity.is_penalized())?;
            crate::cluster_dml_aipw::require_scope(
                query.outcome_functional == antecedent_core::OutcomeFunctional::Mean,
                &query.target_population,
                self.overlap,
            )?;
            crate::cluster_dml_aipw::refuse_interval_request(
                self.bootstrap_replicates,
                self.se_kind,
            )?;
        }
        if let TargetPopulation::CustomDistribution(id) = query.target_population {
            let depends = self.population_registry.as_ref().and_then(|r| r.distribution_dependencies(id))
                .ok_or_else(|| EstimationError::unsupported("AIPW custom weights require declared depends_on via insert_distribution_with_dependence"))?;
            if depends.iter().any(|v| *v == query.treatment || !estimand.adjustment_set.contains(v))
            {
                return Err(EstimationError::unsupported(
                    "AIPW custom weights must depend only on the certified adjustment set",
                ));
            }
        }
        let mut problem = prepare_propensity_problem_with_levels(
            data,
            estimand,
            query,
            self.overlap,
            self.population_registry.as_ref(),
            self.other_levels,
            self.sampling_weights.as_ref(),
        )?;
        problem.propensity = self.propensity.clone();
        if let Some(cluster_dml) = &self.cluster_dml {
            let (first, second) = self.unit_labels(&problem.row_index)?;
            problem.fold_units = Some(cluster_dml.declare_units(
                first.as_deref(),
                second.as_deref(),
                problem.nrows,
            )?);
            problem.fold_unit_kind = cluster_dml.independence_unit();
        }
        Ok(problem)
    }

    /// Sampling weights are read by every mean-functional construction (cross-fitted ATE or
    /// custom target, penalized propensities with a weighted penalty-selection criterion, the
    /// in-sample ATT, ATC and trimmed ATE, and the cluster-DML sandwich). Non-mean functionals
    /// have no sampling-weighted score here and refuse rather than ignore them.
    fn require_weighted_scope(query: &AverageEffectQuery) -> Result<(), EstimationError> {
        if query.outcome_functional != antecedent_core::OutcomeFunctional::Mean {
            return Err(crate::sampling_weights::unsupported_route(
                "AIPW reads sampling weights on the mean functional; a non-mean outcome \
                 functional has no sampling-weighted score here",
            ));
        }
        Ok(())
    }

    /// A penalized propensity is licensed for one construction only: the untrimmed
    /// `AllObserved` mean ATE, whose cross-fitted scores keep the table, row identity,
    /// overlap report and retargetability.
    fn require_penalized_scope(
        &self,
        mean_functional: bool,
        population: &TargetPopulation,
        overlap: OverlapPolicy,
    ) -> Result<(), EstimationError> {
        if !mean_functional
            || !matches!(population, TargetPopulation::AllObserved)
            || trim_of(overlap).is_some()
        {
            // A lasso outside the cross-fit would select on the rows it scores.
            if self.propensity.is_lasso() {
                return Err(PropensityNuisance::lasso_not_cross_fitted());
            }
            return Err(refuse(
                antecedent_core::reason_code!("route_not_supported"),
                "penalized_propensity.scope",
                "a penalized propensity is licensed only for the untrimmed AllObserved mean \
                ATE: other functionals and populations, and trimmed fits, refit full-sample \
                nuisances and have no cross-fitted penalized route",
            ));
        }
        Ok(())
    }

    /// The cluster-DML route publishes no bootstrap, so a request for one is refused instead
    /// of ignored.
    fn require_no_interval_request(&self) -> Result<(), EstimationError> {
        if self.cluster_dml.is_some() {
            crate::cluster_dml_aipw::refuse_interval_request(
                self.bootstrap_replicates,
                self.se_kind,
            )?;
        }
        Ok(())
    }

    /// Prepare the licensed logistic/OLS AIPW route from one identified mean ATE claim.
    ///
    /// The receipt retains the selected functional, identification assumptions, semantic
    /// roles, procedure, overlap policy, uncertainty choices, and complete-case row
    /// identities. This checked route covers the `AllObserved` mean ATE: cross-fitted
    /// scores under an untrimmed policy, the full-sample common-support refit under a
    /// trimming one.
    #[allow(
        clippy::too_many_lines,
        reason = "the checked AIPW receipt binds identification, treatment contrast, program, procedure, and row identities atomically"
    )]
    pub fn prepare_checked(
        &self,
        data: &TabularData,
        identification: &IdentificationResult,
        estimand_index: usize,
    ) -> Result<CheckedAipwPreparation, EstimationError> {
        let crate::prepare::CheckedSelection { estimand: target, claim } =
            crate::prepare::CheckedSelection::resolve(identification, estimand_index)?;
        if claim.status != IdentificationStatus::NonparametricallyIdentified {
            return Err(EstimationError::data_msg(format!(
                "checked AIPW requires a nonparametrically identified claim; got {}",
                claim.status.as_str()
            )));
        }
        let query = identification.average_effect().ok_or_else(|| {
            EstimationError::data_msg("checked AIPW requires an average-effect query")
        })?;
        if !target.is_adjustment_shaped()
            || !target.instruments.is_empty()
            || !target.mediators.is_empty()
        {
            return Err(EstimationError::IncompatibleEstimand {
                message: "checked AIPW requires a back-door adjustment target without instrument or mediator roles",
            });
        }
        crate::prepare::require_unique_roles(
            &target.adjustment_set,
            &[query.treatment, query.outcome],
            "checked AIPW adjustment roles must be unique and exclude treatment and outcome",
        )?;
        if query.outcome_functional != antecedent_core::OutcomeFunctional::Mean
            || query.target_population != TargetPopulation::AllObserved
        {
            return Err(EstimationError::unsupported(
                "checked AIPW currently supports the AllObserved mean ATE",
            ));
        }
        let (active, control, _) =
            crate::prepare::treatment_contrast(&query.active, &query.control)?;
        let mut arena = identification.arena.clone();
        let expected = arena.backdoor_ate(
            query.treatment,
            query.outcome,
            &target.adjustment_set,
            antecedent_core::Value::f64(active),
            antecedent_core::Value::f64(control),
        );
        if expected != target.functional {
            return Err(EstimationError::data_msg(
                "selected AIPW functional does not match treatment, outcome, arms, and adjustment roles",
            ));
        }
        let schema = ProgramSchema::new(
            data.schema()
                .variables()
                .iter()
                .map(|v| (v.id, ProgramVariable { name: Arc::clone(&v.name) })),
        );
        let program = FunctionalProgram::new(
            arena,
            schema,
            target.functional,
            target.functional,
            ProgramLimits::default(),
        )
        .map_err(|e| EstimationError::data_msg(format!("checked AIPW target program: {e}")))?;
        let problem = self.prepare(data, target, query)?;
        if problem.treatment_levels.excluded_rows > 0 {
            return Err(EstimationError::unsupported(
                "checked AIPW targets the all-observed population and cannot exclude treatment rows at undeclared levels",
            ));
        }
        let factor_requirements = Arc::from(program.factor_requirements().to_vec());
        let procedure = CheckedAipwProcedure::for_overlap(self.overlap);
        let lowering = CheckedAipwLowering {
            functional: target.functional,
            treatment: query.treatment,
            outcome: query.outcome,
            adjustment: Arc::clone(&target.adjustment_set),
            population: query.target_population.clone(),
            procedure,
            overlap: self.overlap,
            folds: procedure.folds(),
            se_kind: self.se_kind,
            bootstrap_replicates: self.bootstrap_replicates,
            propensity: self.propensity.clone(),
            cluster_dml: self.cluster_dml,
            rows: Arc::clone(&problem.row_index),
        };
        Ok(CheckedAipwPreparation {
            target: target.clone(),
            program,
            factor_requirements,
            lowering,
            query: query.clone(),
            required_assumptions: claim.required_assumptions.clone(),
            problem,
        })
    }

    /// Rebind a checked AIPW target to compatible data while retaining its selected procedure.
    pub fn rebind_checked(
        &self,
        checked: &CheckedAipwPreparation,
        data: &TabularData,
    ) -> Result<CheckedAipwPreparation, EstimationError> {
        let schema = ProgramSchema::new(
            data.schema()
                .variables()
                .iter()
                .map(|v| (v.id, ProgramVariable { name: Arc::clone(&v.name) })),
        );
        if checked.program.schema() != &schema
            || checked.program.mapping().source != checked.target.functional
            || checked.program.mapping().executable != checked.target.functional
            || checked.lowering.functional != checked.target.functional
            || checked.lowering.treatment != checked.query.treatment
            || checked.lowering.outcome != checked.query.outcome
            || checked.lowering.adjustment != checked.target.adjustment_set
            || checked.lowering.population != checked.query.target_population
            || checked.lowering.procedure != CheckedAipwProcedure::for_overlap(self.overlap)
            || checked.lowering.overlap != self.overlap
            || checked.problem.overlap != self.overlap
            || checked.lowering.folds != checked.lowering.procedure.folds()
            || checked.lowering.se_kind != self.se_kind
            || checked.lowering.bootstrap_replicates != self.bootstrap_replicates
            || checked.lowering.propensity != self.propensity
            || checked.lowering.cluster_dml != self.cluster_dml
        {
            return Err(EstimationError::data_msg(
                "checked AIPW target, procedure, bindings, or semantic schema changed",
            ));
        }
        let mut rebound = checked.clone();
        rebound.problem = self.prepare(data, &checked.target, &checked.query)?;
        rebound.lowering.rows = Arc::clone(&rebound.problem.row_index);
        Ok(rebound)
    }

    /// Execute a checked AIPW preparation using its retained identification assumptions.
    pub fn fit_checked(
        &self,
        checked: &CheckedAipwPreparation,
        workspace: &mut AipwWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<EffectEstimate, EstimationError> {
        if checked.lowering.se_kind != self.se_kind
            || checked.lowering.bootstrap_replicates != self.bootstrap_replicates
            || checked.lowering.propensity != self.propensity
            || checked.lowering.cluster_dml != self.cluster_dml
            || checked.lowering.overlap != self.overlap
            || checked.problem.overlap != self.overlap
            || checked.lowering.procedure != CheckedAipwProcedure::for_overlap(self.overlap)
        {
            return Err(EstimationError::data_msg(
                "AIPW estimator configuration differs from the checked procedure receipt",
            ));
        }
        self.fit(&checked.problem, workspace, ctx, checked.required_assumptions.clone())
    }

    /// Fit propensity + outcome nuisance models and compute the AIPW effect, with optional
    /// bootstrap.
    ///
    /// # Errors
    ///
    /// Target population other than ATE/ATT/ATC, empty treated/control arm, or GLM/OLS
    /// backend failure.
    pub fn fit(
        &self,
        problem: &PreparedPropensityProblem,
        workspace: &mut AipwWorkspace,
        ctx: &ExecutionContext,
        assumptions: AssumptionSet,
    ) -> Result<EffectEstimate, EstimationError> {
        let point = self
            .fit_point(problem, workspace, ctx, assumptions)?
            .with_treatment_levels(problem.treatment_levels);
        self.attach_bootstrap(problem, workspace, ctx, point)
    }

    /// Form a population interval under independent homoskedastic Gaussian arm-linear
    /// outcomes, including the complete training-fold outcome-fit uncertainty.
    ///
    /// This explicit model construction leaves the robust default fit unchanged. Its
    /// Gaussian conditional noise term and asymptotic random-population term are distinct.
    ///
    /// # Errors
    /// Unsupported dependence, population, trimming, failed fit, or invalid confidence level.
    pub fn population_linear_interval(
        &self,
        problem: &PreparedPropensityProblem,
        workspace: &mut AipwWorkspace,
        ctx: &ExecutionContext,
        assumptions: AssumptionSet,
        level: f64,
    ) -> Result<crate::population_aipw::PopulationAipwInterval, EstimationError> {
        if self.cluster_dml.is_some()
            || self.cluster_ids.is_some()
            || self.source_cluster_ids.is_some()
            || self.multiway_ids.is_some()
            || !matches!(self.se_kind, AnalyticSeKind::Homoskedastic | AnalyticSeKind::Hc1)
            || self.bootstrap_replicates != 0
        {
            return Err(EstimationError::unsupported(
                "population linear AIPW requires independent rows without bootstrap",
            ));
        }
        let estimate = self.fit(problem, workspace, ctx, assumptions)?;
        let table = estimate.scores.table.as_ref().ok_or_else(|| {
            EstimationError::unsupported(
                "population linear AIPW requires its cross-fit score table",
            )
        })?;
        let propensity = table.propensities[problem.nrows..2 * problem.nrows].to_vec();
        let folds = table.fold_ids.to_vec();
        crate::population_aipw::interval(problem, estimate, &propensity, &folds, 0.0, level)
    }

    /// Form a control-population interval under independent arm-linear conditional means
    /// with arbitrary heteroskedastic outcome variances. The complete treated-OLS training
    /// coefficient enters an HC3 noise variance, with separate control-target heterogeneity.
    /// This explicit asymptotic model construction leaves the default ATC fit unchanged.
    ///
    /// # Errors
    /// Dependence, trimming, a non-control target, invalid level, or failed nuisance fit.
    pub fn population_linear_atc_interval(
        &self,
        problem: &PreparedPropensityProblem,
        workspace: &mut AipwWorkspace,
        ctx: &ExecutionContext,
        assumptions: AssumptionSet,
        level: f64,
    ) -> Result<crate::population_atc::PopulationAtcInterval, EstimationError> {
        if self.cluster_dml.is_some()
            || self.cluster_ids.is_some()
            || self.source_cluster_ids.is_some()
            || self.multiway_ids.is_some()
            || !matches!(self.se_kind, AnalyticSeKind::Homoskedastic | AnalyticSeKind::Hc1)
            || self.bootstrap_replicates != 0
        {
            return Err(EstimationError::unsupported(
                "arm-linear ATC requires independent rows without bootstrap",
            ));
        }
        let estimate = self.fit(problem, workspace, ctx, assumptions)?;
        let model = PropensityModel::fit(
            problem,
            self.backend,
            &mut workspace.propensity,
            &self.glm_options,
        )?;
        crate::population_atc::interval(problem, estimate, &model.clipped_scores, level)
    }

    /// Attach the bootstrap SE onto a point estimate from [`Self::fit`] (progressive
    /// uncertainty stage), without refitting the nuisance models. The resampling scheme is
    /// the one the point fit's route implies: cross-fitted score-table refits for custom
    /// weights and the untrimmed ATE, per-replicate nuisance refits otherwise.
    ///
    /// A penalized propensity (declared, or the destination of a fallback that ran) repeats the
    /// fold plan, the penalty (and lasso support) selection and every nuisance fit on each
    /// resample, and records the penalties each replicate selected in
    /// [`crate::EstimateDiagnostics::penalized`]. When the point came from a fallback, the replicates run
    /// the destination (the claim is the destination's); when it came from the GLM, they run
    /// the GLM without the fallback, so a failing replicate is counted rather than replaced.
    ///
    /// # Errors
    ///
    /// Bootstrap failure.
    pub fn attach_bootstrap(
        &self,
        problem: &PreparedPropensityProblem,
        workspace: &mut AipwWorkspace,
        ctx: &ExecutionContext,
        point: EffectEstimate,
    ) -> Result<EffectEstimate, EstimationError> {
        self.require_no_interval_request()?;
        if self.bootstrap_replicates == 0 {
            return Ok(point);
        }
        let boot = if matches!(problem.target_population, TargetPopulation::CustomDistribution(_)) {
            let weights = problem
                .target_weights
                .as_deref()
                .ok_or_else(|| EstimationError::data_msg("missing custom target weights"))?;
            self.crossfit_bootstrap(problem, ctx, Some(weights))?
        } else if matches!(problem.target_population, TargetPopulation::AllObserved)
            && trim_of(problem.overlap).is_none()
        {
            let nuisance = self.replicate_nuisance(&point);
            if nuisance.is_penalized() {
                return self.attach_penalized_bootstrap(problem, ctx, nuisance, point);
            }
            self.crossfit_bootstrap(problem, ctx, None)?
        } else {
            self.bootstrap_se(problem, workspace, ctx)?
        };
        Ok(point.with_bootstrap_outcome(boot))
    }

    #[allow(
        clippy::too_many_lines,
        reason = "one point fit in order: check the route, fit the propensity, trim, fit both outcome arms, then score and assemble; every stage reads the retained rows"
    )]
    fn fit_point(
        &self,
        problem: &PreparedPropensityProblem,
        workspace: &mut AipwWorkspace,
        ctx: &ExecutionContext,
        assumptions: AssumptionSet,
    ) -> Result<EffectEstimate, EstimationError> {
        if problem.propensity != self.propensity {
            return Err(EstimationError::data_msg(
                "AIPW propensity nuisance differs from the one the problem was prepared with",
            ));
        }
        self.propensity.validate_for_execution()?;
        if self.propensity.is_penalized() {
            self.require_penalized_scope(true, &problem.target_population, problem.overlap)?;
        }
        if self.cluster_dml.is_some() {
            crate::cluster_dml_aipw::require_unpenalized(self.propensity.is_penalized())?;
            crate::cluster_dml_aipw::require_scope(
                true,
                &problem.target_population,
                problem.overlap,
            )?;
            crate::cluster_dml_aipw::refuse_interval_request(
                self.bootstrap_replicates,
                self.se_kind,
            )?;
        }
        let declared_units = self.cluster_dml.and_then(|spec| {
            let (first, second) = self.unit_labels(&problem.row_index).ok()?;
            spec.declare_units(first.as_deref(), second.as_deref(), problem.nrows).ok()
        });
        if problem.fold_units.as_deref() != declared_units.as_deref()
            || self
                .cluster_dml
                .is_some_and(|spec| spec.independence_unit() != problem.fold_unit_kind)
        {
            return Err(EstimationError::data_msg(
                "AIPW cluster-DML declaration differs from the one the problem was prepared with",
            ));
        }
        if !matches!(
            problem.target_population,
            TargetPopulation::AllObserved
                | TargetPopulation::Treated
                | TargetPopulation::Untreated
                | TargetPopulation::Predicate(_)
                | TargetPopulation::CustomDistribution(_)
        ) {
            return Err(EstimationError::unsupported(
                "AIPW supports AllObserved, Treated, Untreated, Predicate, or CustomDistribution",
            ));
        }

        if matches!(problem.target_population, TargetPopulation::CustomDistribution(_)) {
            return self.fit_custom(problem, ctx, assumptions);
        }
        if matches!(problem.target_population, TargetPopulation::AllObserved)
            && trim_of(problem.overlap).is_none()
        {
            return self.fit_crossfit_scores(
                problem,
                ctx,
                assumptions,
                problem.score_weights().as_deref(),
                false,
            );
        }
        let model = PropensityModel::fit(
            problem,
            self.backend,
            &mut workspace.propensity,
            &self.glm_options,
        )
        .map_err(|error| self.propensity.record_failed_fit(error))?;
        // Trim on raw scores: excluded units match the overlap report's common-support claim.
        let retained = trim_retained_rows(&model.fit.scores, trim_of(problem.overlap))?;
        let ncols = problem.design_ncols;
        // Sampling weights of the rows the contrast uses, renormalized to mean one among
        // them (a trimmed fit estimates the effect in the retained weighted population).
        let used_sampling = match (&retained, problem.sampling_weights.as_ref()) {
            (Some(idx), Some(sampling)) => Some(sampling.subset(idx)),
            (None, Some(sampling)) => Some(sampling.clone()),
            (_, None) => None,
        };
        let used_weights: Option<&[f64]> = used_sampling.as_ref().map(|s| &*s.weights);
        let (design_used, t_used, y_used, e_used) = match &retained {
            Some(idx) => {
                let mut design = Vec::new();
                select_rows_colmajor(
                    &problem.design_matrix,
                    problem.nrows,
                    ncols,
                    idx,
                    &mut design,
                );
                (
                    Cow::Owned(design),
                    Cow::Owned(gather(&problem.treatment, idx)),
                    Cow::Owned(gather(&problem.outcome, idx)),
                    Cow::Owned(gather(&model.clipped_scores, idx)),
                )
            }
            None => (
                Cow::Borrowed(problem.design_matrix.as_ref()),
                Cow::Borrowed(problem.treatment.as_ref()),
                Cow::Borrowed(problem.outcome.as_ref()),
                Cow::Borrowed(model.clipped_scores.as_slice()),
            ),
        };
        let nrows = t_used.len();
        let (beta0, beta1) = fit_outcome_models_weighted(
            &design_used,
            nrows,
            ncols,
            &t_used,
            &y_used,
            used_weights,
            self.backend,
            workspace,
        )?;
        predict_colmajor(&design_used, nrows, ncols, &beta0, &mut workspace.mu0);
        predict_colmajor(&design_used, nrows, ncols, &beta1, &mut workspace.mu1);
        aipw_psi_weighted(
            &t_used,
            &y_used,
            &e_used,
            &workspace.mu0,
            &workspace.mu1,
            &problem.target_population,
            used_weights,
            &mut workspace.psi,
        )?;
        if let Some(weights) = used_weights {
            // The weighted estimate is the mean of `m_i ψ_i`; every later step (centering,
            // nuisance correction, variance) reads these weighted scores.
            for (value, &m) in workspace.psi.iter_mut().zip(weights) {
                *value *= m;
            }
        }
        let ate = workspace.psi.iter().sum::<f64>() / workspace.psi.len() as f64;
        if matches!(
            problem.target_population,
            TargetPopulation::Treated | TargetPopulation::Untreated
        ) {
            // ATT/ATC: when both nuisances are consistent, the centered DR score
            // is the efficient influence function (Hahn 1998); under one-model
            // misspecification it omits nuisance terms, so this SE is not doubly
            // robust even though the point estimate is. The estimand depends on the propensity
            // (it averages over the treated / control law), so E[ψ·S_γ] ≠ 0 and
            // projecting ψ off the logistic scores would remove genuine variance.
            center_population_psi(
                &mut workspace.psi,
                &t_used,
                &problem.target_population,
                ate,
                used_weights,
            );
        } else {
            let e_raw: Cow<'_, [f64]> = match &retained {
                Some(idx) => Cow::Owned(gather(&model.fit.scores, idx)),
                None => Cow::Borrowed(model.fit.scores.as_slice()),
            };
            let fit = AipwNuisanceFit {
                treatment: &t_used,
                outcome: &y_used,
                e_used: &e_used,
                e_raw: &e_raw,
                mu0: &workspace.mu0,
                mu1: &workspace.mu1,
                design: &design_used,
                ncols,
                weights: used_weights,
            };
            correct_aipw_psi_for_nuisances(&mut workspace.psi, &fit)?;
        }
        let declared_clusters = crate::se::declared_row_labels(
            self.cluster_ids.as_deref(),
            self.source_cluster_ids.as_deref(),
            problem.row_index.iter().map(|row| usize::try_from(row.get()).expect("row id")),
        )?;
        let se_analytic = crate::se::influence_se_kind(
            self.se_kind,
            &workspace.psi,
            problem.nrows,
            declared_clusters.as_deref(),
            self.multiway_ids.as_deref(),
            self.panel_times.as_deref(),
            retained.as_deref(),
        )?;
        let overlap_report = Some(crate::propensity::propensity_overlap_report(
            problem,
            &model.fit.scores,
            None,
            IpwTarget::from_population(&problem.target_population).ok(),
        ));
        let estimate = EffectEstimate::new(ate, se_analytic, assumptions, problem.overlap)
            .with_preparation(Some(problem.preparation_summary()))
            .with_se_kind(self.se_kind)
            .with_sampling_weights(used_sampling.as_ref().map(|s| s.record))
            .with_overlap_report(overlap_report)
            .with_influence(Some(Arc::from(workspace.psi.as_slice())));
        Ok(estimate)
    }

    fn fit_custom(
        &self,
        problem: &PreparedPropensityProblem,
        ctx: &ExecutionContext,
        assumptions: AssumptionSet,
    ) -> Result<EffectEstimate, EstimationError> {
        if trim_of(problem.overlap).is_some()
            || !matches!(self.se_kind, AnalyticSeKind::Homoskedastic)
        {
            return Err(EstimationError::unsupported(
                "cross-fitted custom AIPW currently requires iid rows and no data-dependent trimming",
            ));
        }
        let weights = problem
            .score_weights()
            .ok_or_else(|| EstimationError::data_msg("missing custom target weights"))?;
        self.fit_crossfit_scores(problem, ctx, assumptions, Some(&weights), true)
    }

    /// Cross-fitted binary AIPW contrast. Uniform weights (`None`) are the
    /// `AllObserved` retarget(`1`) object.
    #[expect(
        clippy::too_many_lines,
        reason = "one linear cross-fit pass: nuisances, scores, influence correction and variance"
    )]
    fn fit_crossfit_scores(
        &self,
        problem: &PreparedPropensityProblem,
        ctx: &ExecutionContext,
        assumptions: AssumptionSet,
        weights: Option<&[f64]>,
        refuse_overlap: bool,
    ) -> Result<EffectEstimate, EstimationError> {
        let fold_seed = ctx.rng.master_seed();
        let (table, nuisance) = self.crossfit_table(problem, true, ctx, Some(fold_seed))?;
        let crate::crossfit_aipw::NuisanceFit { selections, fallback } = nuisance;
        let n_folds = table.n_folds as usize;
        let report = PenalizedReport {
            fallback,
            selected_support: fold_supports(&selections, &problem.adjustment_set),
            variance: None,
        };
        let penalized_route = self.propensity.is_penalized() || report.fallback.is_some();
        let penalized_provenance: Vec<_> = if penalized_route {
            selections.iter().map(|s| s.provenance.clone()).collect()
        } else {
            Vec::new()
        };
        let summary = table.summarize(weights)?;
        let contrast = table.linear_contrast(&summary, &[-1.0, 1.0])?;
        let iid_se = matches!(self.se_kind, AnalyticSeKind::Homoskedastic);
        let influence = contrast_influence(&table, weights, contrast.value)?;
        // The score table's covariance and simultaneous bands are iid objects; under a
        // dependence-robust SE kind they would contradict `se_analytic`, so they are withheld
        // rather than published at the wrong strength.
        let inference = table.inference(weights)?;
        if refuse_overlap && !inference.support.overlap_ok {
            return Err(EstimationError::unsupported_with_fields(
                "custom target weighted overlap failed",
                crate::retarget::weighted_overlap_fields(
                    &table,
                    weights.unwrap_or(&[]),
                    &inference.support,
                ),
            ));
        }
        let se_analytic = if iid_se {
            contrast.se
        } else {
            let declared_clusters = crate::se::declared_row_labels(
                self.cluster_ids.as_deref(),
                self.source_cluster_ids.as_deref(),
                problem.row_index.iter().map(|row| usize::try_from(row.get()).expect("row id")),
            )?;
            crate::se::influence_se_kind(
                self.se_kind,
                &influence,
                problem.nrows,
                declared_clusters.as_deref(),
                self.multiway_ids.as_deref(),
                self.panel_times.as_deref(),
                None,
            )?
        };
        let overlap_report = treated_arm_overlap_report(&table, problem);
        // Whole-cluster folds: the sandwich standard error of the same scores, under the declared
        // independence unit (`docs/guides/clustered-dml.md`).
        let (first_units, second_units) = if self.cluster_dml.is_some() {
            self.unit_labels(&problem.row_index)?
        } else {
            (None, None)
        };
        let cluster_se = match (self.cluster_dml, first_units.as_deref()) {
            (Some(spec), Some(labels)) => {
                let receipt = spec.receipt_with_influence(
                    &table,
                    labels,
                    second_units.as_deref(),
                    problem.sampling_weights.is_some().then_some(&influence[..]),
                )?;
                Some((receipt.cluster_sandwich_se, receipt.reference_df))
            }
            _ => None,
        };
        let mut result =
            EffectEstimate::new(contrast.value, se_analytic, assumptions, problem.overlap)
                .with_preparation(Some(problem.preparation_summary()))
                .with_se_kind(self.se_kind)
                .with_sampling_weights(problem.sampling_weights.as_ref().map(|s| s.record))
                .with_overlap_report(overlap_report)
                .with_joint_covariance(iid_se.then_some(summary.covariance))
                .with_influence(Some(influence.into()))
                .with_score_table(Some(table));
        result.scores.inference = iid_se.then_some(inference);
        if penalized_route {
            // The penalized route publishes the same cross-fitted scores, influence values and
            // iid / robust SE as the unpenalized one (the interval rests on the remainder
            // condition of `docs/guides/penalized-aipw.md`) and records which nuisances it
            // selected, which fold selected them, and any fallback that replaced a failed fit.
            result.crossfit.folds = Some(n_folds);
            result.crossfit.seed = Some(fold_seed);
            result.crossfit.learner_provenance = penalized_provenance;
        }
        if !report.is_empty() {
            result.diagnostics.penalized = Some(Box::new(report));
        }
        if self.cluster_dml.is_some() {
            // The cluster-DML route publishes the cluster-sandwich SE below; the iid
            // covariance and influence values of these scores are not a licensed sampling
            // distribution for it (see `docs/guides/clustered-dml.md`).
            result.uncertainty.analytic.se = f64::NAN;
            result.uncertainty.analytic.kind = None;
            result.joint.covariance = None;
            result.scores.inference = None;
            result.scores.influence = None;
            result.crossfit.folds = Some(n_folds);
            result.crossfit.seed = Some(fold_seed);
            result.crossfit.learner_provenance =
                selections.into_iter().map(|s| s.provenance).collect();
            if let Some((se, df)) = cluster_se {
                // The cluster sandwich replaces the iid standard error; the iid joint
                // covariance and influence values stay withheld (wrong dependence).
                result.uncertainty.analytic.se = se;
                result.uncertainty.analytic.reference_df =
                    Some(f64::from(u32::try_from(df).unwrap_or(u32::MAX)));
                result.uncertainty.analytic.kind = Some(if self.has_second_units() {
                    AnalyticSeKind::Multiway
                } else {
                    AnalyticSeKind::Cluster
                });
            }
        }
        Ok(result)
    }

    /// The recorded cross-fit seed must govern the fold plan, not only the learners.
    ///
    /// Production fits pass that seed into the score table. The tests still build the
    /// seeded problem so a resample can be compared with the original.
    #[cfg(test)]
    fn crossfit_problem(
        problem: &PreparedPropensityProblem,
        ctx: &ExecutionContext,
    ) -> PreparedPropensityProblem {
        let mut seeded = problem.clone();
        seeded.fold_seed = ctx.rng.master_seed();
        seeded
    }

    /// `share`: whether a batch's shared nuisance cache may serve this table (the
    /// point fit) or not (bootstrap replicates).
    /// A penalized propensity's per-fold selections, and any fallback that ran, come back
    /// beside the table.
    fn crossfit_table(
        &self,
        p: &PreparedPropensityProblem,
        share: bool,
        ctx: &ExecutionContext,
        fold_seed: Option<u64>,
    ) -> Result<(crate::scores::ScoreTable, crate::crossfit_aipw::NuisanceFit), EstimationError>
    {
        crate::crossfit_aipw::build_binary_scores_in(
            p,
            &crate::crossfit_aipw::ScoreBuild {
                treatment: p.treatment_id,
                thresholds: &[None],
                folds: crate::crossfit_aipw::DEFAULT_AIPW_FOLDS,
                glm_options: &self.glm_options,
                backend: self.backend,
                share,
                ctx: Some(ctx),
                fold_seed,
            },
        )
    }

    /// Row-resample bootstrap of the cross-fitted contrast: each replicate rebuilds the score
    /// table on the resampled rows. `weights` are the custom-target weights (`None` = uniform).
    fn crossfit_bootstrap(
        &self,
        problem: &PreparedPropensityProblem,
        ctx: &ExecutionContext,
        weights: Option<&[f64]>,
    ) -> Result<BootstrapOutcome, EstimationError> {
        let fold_seed = ctx.rng.master_seed();
        let clusters = self.bootstrap_clusters(problem)?;
        let series_block = self.series_block(problem.nrows);
        // Each replicate clones the problem with a resampled design and four row vectors.
        let _gather = crate::util::admit_gather_scratch(
            ctx,
            problem.nrows,
            problem.design_ncols + 4,
            self.bootstrap_replicates,
        )?;
        crate::util::bootstrap_outcome_by_design(
            self.bootstrap_interval.as_ref(),
            crate::util::BootstrapDesign {
                clusters: clusters.as_deref(),
                series_block,
                declared: self.sampling_design,
            },
            self.bootstrap_replicates,
            ctx,
            0xA1D5,
            problem.nrows,
            || (),
            |(), idx| {
                let mut p = resampled_problem(problem, idx, weights);
                // The point is the GLM result: a replicate whose GLM fit fails is a failed
                // replicate, never silently a ridge fit.
                p.propensity = PropensityNuisance::default();
                let Ok((t, _)) = self.crossfit_table(&p, false, ctx, Some(fold_seed)) else {
                    return Ok(None);
                };
                let ss = t.summarize(p.score_weights().as_deref())?;
                Ok(Some(ss.means[1] - ss.means[0]))
            },
        )
    }

    /// The nuisance configuration a bootstrap replicate runs: the fallback destination when
    /// the point came from a fallback, the declared penalized propensity, or the plain GLM.
    fn replicate_nuisance(&self, point: &EffectEstimate) -> PropensityNuisance {
        let fell_back =
            point.diagnostics.penalized.as_ref().is_some_and(|report| report.fallback.is_some());
        if fell_back {
            self.propensity.fallback_destination().unwrap_or_default()
        } else if self.propensity.is_penalized() {
            self.propensity.clone()
        } else {
            PropensityNuisance::default()
        }
    }

    /// Unit-resample bootstrap of the penalized cross-fitted contrast.
    ///
    /// Every replicate draws `n` rows with replacement into a reused buffer (copies keep their
    /// unit identity, so a unit is never in a training set and in its own validation fold) and
    /// re-runs the whole pipeline on the resample under `nuisance`: the arm-stratified unit
    /// fold plan, each fold's inner-CV penalty (and lasso support) selection and every nuisance
    /// fit. Replicate `r` is drawn from [`antecedent_core::StreamDomain::Resample`] at
    /// `derive_stream(PENALIZED_BOOTSTRAP_STREAM, r)`, so it is fixed by the seed and `r` alone.
    /// The penalties each successful replicate selected are recorded in replicate order, so the
    /// record replays from the seed under any thread count. A replicate that fails softly
    /// (a fold missing an arm, a failed fit, a boundary propensity) is counted, never
    /// replaced; a cancelled one stops the loop and is reported as a stop.
    fn attach_penalized_bootstrap(
        &self,
        problem: &PreparedPropensityProblem,
        ctx: &ExecutionContext,
        nuisance: PropensityNuisance,
        point: EffectEstimate,
    ) -> Result<EffectEstimate, EstimationError> {
        let base = penalized_bootstrap_base(problem, ctx.rng.master_seed(), nuisance);
        let n = base.nrows;
        let n_rep = self.bootstrap_replicates as usize;
        if let Some(request) = self.bootstrap_interval {
            return self.attach_penalized_waves(&base, ctx, request, point);
        }
        let outcomes = if ctx.cancellation.is_cancelled() {
            vec![PenalizedDraw::Cancelled]
        } else if n == 0 || n_rep == 0 {
            return Err(EstimationError::Data(antecedent_data::DataError::InvalidArgument {
                message: "batch resample needs n > 0 and n_replicates > 0".into(),
            }));
        } else {
            let threads = (ctx.parallelism.max_threads.get() as usize).clamp(1, n_rep);
            retained_index_slots(n, threads)?;
            if u64::try_from(n.saturating_sub(1)).is_err() {
                return Err(EstimationError::data_msg(
                    "address the row by GlobalId and keep ChunkIndex inside one bounded chunk. A u32 backend feature flag is not that attempt",
                ));
            }
            map_penalized_replicates(ctx, n_rep, |r, inner, ids, idx| {
                self.penalized_replicate(&base, ctx, inner, n, r, (ids, idx))
            })?
        };
        let mut ates = Vec::with_capacity(n_rep);
        let mut penalties = Vec::with_capacity(n_rep);
        let (mut attempted, mut cancelled) = (0u32, false);
        for (r, outcome) in outcomes.into_iter().enumerate() {
            match outcome {
                PenalizedDraw::Done(ate, lambdas) => {
                    attempted += 1;
                    ates.push(ate);
                    penalties.push(ReplicatePenalties {
                        replicate: u32::try_from(r).unwrap_or(u32::MAX),
                        lambdas,
                    });
                }
                PenalizedDraw::Failed => attempted += 1,
                PenalizedDraw::Cancelled => cancelled = true,
            }
        }
        let boot = crate::util::finalize_bootstrap_se_ex(&ates, attempted, cancelled, false);
        let influence_se = match point.scores.table.as_ref() {
            Some(table) => crossfit_influence_se(table)?,
            None => f64::NAN,
        };
        let mut report = point.diagnostics.penalized.as_deref().cloned().unwrap_or_default();
        report.variance = Some(PenalizedVarianceReport {
            uncertainty_kind: REFIT_BOOTSTRAP_UNCERTAINTY_KIND,
            refit_bootstrap_se: boot.se,
            replicates_requested: self.bootstrap_replicates,
            replicates_ok: boot.replicates_ok,
            replicates_failed: boot.replicates_failed,
            cancelled: boot.cancelled,
            replicate_penalties: penalties,
            influence_se,
        });
        let mut estimate = point.with_bootstrap(Some(boot));
        estimate.diagnostics.penalized = Some(Box::new(report));
        Ok(estimate)
    }

    /// The penalized refit bootstrap as the wave procedure of a requested interval.
    ///
    /// Replicate `r` is the same penalized refit on the same rows as in the fixed-count loop
    /// (stream `derive_stream(PENALIZED_BOOTSTRAP_STREAM, r)`); waves run on the caller and stop
    /// on the request's Monte Carlo tolerances, capped by the declared replicate count. The
    /// penalties the kept replicates selected are recorded in replicate order, and the report
    /// of the procedure travels beside them.
    fn attach_penalized_waves(
        &self,
        base: &PreparedPropensityProblem,
        ctx: &ExecutionContext,
        request: crate::util::IntervalRequest,
        point: EffectEstimate,
    ) -> Result<EffectEstimate, EstimationError> {
        use crate::adaptive_resample_waves::{DriveError, run_waves_serial};
        let n = base.nrows;
        if n == 0 || self.bootstrap_replicates == 0 {
            return Err(EstimationError::Data(antecedent_data::DataError::InvalidArgument {
                message: "batch resample needs n > 0 and n_replicates > 0".into(),
            }));
        }
        retained_index_slots(n, 1)?;
        let rule = request.rule(self.bootstrap_replicates);
        let mut state = (Vec::<antecedent_data::GlobalId>::new(), Vec::<usize>::new(), Vec::new());
        let run = run_waves_serial(
            &rule,
            ctx,
            PENALIZED_BOOTSTRAP_STREAM,
            &mut state,
            |(ids, idx, penalties): &mut (_, _, Vec<ReplicatePenalties>), id, _rng| {
                let replicate = usize::try_from(id).map_err(|_| {
                    EstimationError::data_msg("bootstrap replicate id exceeds usize")
                })?;
                Ok(match self.penalized_replicate(base, ctx, ctx, n, replicate, (ids, idx))? {
                    PenalizedDraw::Done(ate, lambdas) => {
                        penalties.push(ReplicatePenalties {
                            replicate: u32::try_from(id).unwrap_or(u32::MAX),
                            lambdas,
                        });
                        Some(ate)
                    }
                    // A cancelled draw is observed again before the next replicate, which
                    // discards the wave it belongs to.
                    PenalizedDraw::Failed | PenalizedDraw::Cancelled => None,
                })
            },
        );
        let run = match run {
            Ok(run) => run,
            Err(DriveError::Estimator(error)) => return Err(error),
            Err(DriveError::Wave(error)) => {
                return Err(EstimationError::data_msg(error.to_string()));
            }
        };
        // A wave discarded by cancellation leaves its penalties behind: keep exactly the
        // replicates whose estimates the run kept.
        let mut penalties = state.2;
        penalties.truncate(run.successes.len());
        let wave = crate::util::WaveBootstrap::from_run(run);
        let boot = wave.se;
        let influence_se = match point.scores.table.as_ref() {
            Some(table) => crossfit_influence_se(table)?,
            None => f64::NAN,
        };
        let mut report = point.diagnostics.penalized.as_deref().cloned().unwrap_or_default();
        report.variance = Some(PenalizedVarianceReport {
            uncertainty_kind: REFIT_BOOTSTRAP_UNCERTAINTY_KIND,
            refit_bootstrap_se: boot.se,
            replicates_requested: self.bootstrap_replicates,
            replicates_ok: boot.replicates_ok,
            replicates_failed: boot.replicates_failed,
            cancelled: boot.cancelled,
            replicate_penalties: penalties,
            influence_se,
        });
        let resampling = wave.report(rule);
        let mut estimate = point.with_bootstrap(Some(boot));
        estimate.diagnostics.penalized = Some(Box::new(report));
        estimate.diagnostics.resampling = Some(resampling);
        Ok(estimate)
    }

    /// One penalized refit on the rows drawn for replicate `r`.
    ///
    /// Indexes are written into the caller's buffers. `outer` is the stream the batch used;
    /// `inner` is the context that evaluates the replicate (serial when the pool is parallel).
    fn penalized_replicate(
        &self,
        base: &PreparedPropensityProblem,
        outer: &ExecutionContext,
        inner: &ExecutionContext,
        n: usize,
        replicate: usize,
        indexes: (&mut Vec<antecedent_data::GlobalId>, &mut Vec<usize>),
    ) -> Result<PenalizedDraw, EstimationError> {
        let (ids, idx) = indexes;
        if inner.cancellation.is_cancelled() {
            return Ok(PenalizedDraw::Cancelled);
        }
        let replicate = u64::try_from(replicate)
            .map_err(|_| EstimationError::data_msg("bootstrap replicate id exceeds u64"))?;
        let clusters = self.bootstrap_clusters(base)?;
        let series_block = self.series_block(n);
        match fill_penalized_bootstrap_indexes(
            n,
            replicate,
            outer,
            ids,
            idx,
            clusters.as_deref(),
            series_block,
            self.sampling_design,
        ) {
            Ok(()) => {}
            Err(EstimationError::Data(antecedent_data::DataError::Cancelled)) => {
                return Ok(PenalizedDraw::Cancelled);
            }
            Err(error) => return Err(error),
        }
        let p = resampled_problem(base, idx, None);
        Ok(match self.crossfit_table(&p, false, inner, None) {
            Ok((table, fit)) => match table.summarize(None) {
                Ok(summary) => {
                    let ate = summary.means[1] - summary.means[0];
                    if ate.is_finite() {
                        let lambdas = fit.selections.iter().map(|s| s.lambda).collect();
                        PenalizedDraw::Done(ate, lambdas)
                    } else {
                        PenalizedDraw::Failed
                    }
                }
                Err(_) => PenalizedDraw::Failed,
            },
            Err(_) if inner.cancellation.is_cancelled() => PenalizedDraw::Cancelled,
            Err(_) => PenalizedDraw::Failed,
        })
    }

    fn bootstrap_clusters(
        &self,
        problem: &PreparedPropensityProblem,
    ) -> Result<Option<std::borrow::Cow<'_, [u32]>>, EstimationError> {
        let labels = crate::se::declared_row_labels(
            self.cluster_ids.as_deref(),
            self.source_cluster_ids.as_deref(),
            problem.row_index.iter().map(|row| usize::try_from(row.get()).expect("row id")),
        )?;
        if let Some(labels) = &labels {
            crate::se::require_clusters(Some(labels.as_ref()), problem.nrows)?;
        }
        Ok(labels)
    }

    fn series_block(&self, n: usize) -> Option<usize> {
        self.panel_times.as_ref().map(|_| {
            let span = match self.se_kind {
                AnalyticSeKind::NeweyWest { lag } | AnalyticSeKind::PanelClusterHac { lag } => {
                    lag.max(1)
                }
                _ => 1,
            };
            antecedent_data::circular_block_length(span, n)
        })
    }

    #[expect(
        clippy::too_many_lines,
        reason = "one linear replicate loop: resample, refit nuisances, rescore and aggregate"
    )]
    fn bootstrap_se(
        &self,
        problem: &PreparedPropensityProblem,
        _workspace: &mut AipwWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<BootstrapOutcome, EstimationError> {
        let clip = clip_of(problem.overlap);
        let trim = trim_of(problem.overlap);
        let n = problem.nrows;
        let ncols = problem.design_ncols;
        let clusters = self.bootstrap_clusters(problem)?;
        let series_block = self.series_block(n);
        // Each worker gathers the design, three vectors, and four trimmed copies.
        let _gather =
            crate::util::admit_gather_scratch(ctx, n, ncols + 8, self.bootstrap_replicates)?;
        // Replicates own a cold workspace; parallel workers cannot share the caller's.
        crate::util::bootstrap_outcome_by_design(
            self.bootstrap_interval.as_ref(),
            crate::util::BootstrapDesign {
                clusters: clusters.as_deref(),
                series_block,
                declared: self.sampling_design,
            },
            self.bootstrap_replicates,
            ctx,
            0xA1D0_u64,
            n,
            || {
                (
                    AipwWorkspace::default(),
                    vec![0.0; n * ncols],
                    vec![0.0; n],
                    vec![0.0; n],
                    vec![0.0; n],
                    Vec::<f64>::new(),
                    Vec::<f64>::new(),
                    Vec::<f64>::new(),
                    Vec::<f64>::new(),
                )
            },
            |(workspace, x_boot, t_boot, y_boot, e, design_trim, t_trim, y_trim, e_trim), idx| {
                let drawn = idx.len();
                x_boot.resize(drawn * ncols, 0.0);
                for buffer in [&mut *t_boot, &mut *y_boot, &mut *e] {
                    buffer.resize(drawn, 0.0);
                }
                crate::util::gather_bootstrap_vector(t_boot, &problem.treatment, idx);
                crate::util::gather_bootstrap_vector(y_boot, &problem.outcome, idx);
                crate::util::gather_cols(x_boot, &problem.design_matrix, n, ncols, idx);
                // The resample carries each drawn row's sampling weight.
                let w_boot: Option<Vec<f64>> = problem
                    .sampling_weights
                    .as_ref()
                    .map(|s| idx.iter().map(|&i| s.weights[i]).collect());
                let propensity_fit = match w_boot.as_deref() {
                    Some(w) => antecedent_stats::fit_propensity_weighted_in_place(
                        x_boot,
                        drawn,
                        ncols,
                        t_boot,
                        w,
                        &self.backend,
                        &mut workspace.propensity,
                        &self.glm_options,
                    ),
                    None => antecedent_stats::fit_propensity_in_place(
                        x_boot,
                        drawn,
                        ncols,
                        t_boot,
                        &self.backend,
                        &mut workspace.propensity,
                        &self.glm_options,
                    ),
                };
                if propensity_fit.is_err() {
                    return Ok(None);
                }
                let raw = &workspace.propensity.scores[..drawn];
                e.copy_from_slice(raw);
                if let Some(c) = clip {
                    clamp_scores(e, c);
                }
                let Ok(retained) = trim_retained_rows(raw, trim) else {
                    return Ok(None);
                };
                let (design_used, t_used, y_used, e_used): (&[f64], &[f64], &[f64], &[f64]) =
                    match &retained {
                        Some(kept) => {
                            select_rows_colmajor(x_boot, drawn, ncols, kept, design_trim);
                            gather_into(t_trim, t_boot, kept);
                            gather_into(y_trim, y_boot, kept);
                            gather_into(e_trim, e, kept);
                            (design_trim, t_trim, y_trim, e_trim)
                        }
                        None => (x_boot, t_boot, y_boot, e),
                    };
                let nrows = t_used.len();
                // Weights of the rows the contrast uses, renormalized to mean one.
                let w_used: Option<Vec<f64>> = w_boot.as_ref().map(|w| {
                    let mut used: Vec<f64> = match &retained {
                        Some(rows) => rows.iter().map(|&i| w[i]).collect(),
                        None => w.clone(),
                    };
                    let scale = used.len() as f64 / used.iter().sum::<f64>();
                    for value in &mut used {
                        *value *= scale;
                    }
                    used
                });
                let Ok((beta0, beta1)) = fit_outcome_models_weighted(
                    design_used,
                    nrows,
                    ncols,
                    t_used,
                    y_used,
                    w_used.as_deref(),
                    self.backend,
                    workspace,
                ) else {
                    return Ok(None);
                };
                predict_colmajor(design_used, nrows, ncols, &beta0, &mut workspace.mu0);
                predict_colmajor(design_used, nrows, ncols, &beta1, &mut workspace.mu1);
                if aipw_psi_weighted(
                    t_used,
                    y_used,
                    e_used,
                    &workspace.mu0,
                    &workspace.mu1,
                    &problem.target_population,
                    w_used.as_deref(),
                    &mut workspace.psi,
                )
                .is_err()
                {
                    return Ok(None);
                }
                if let Some(w) = w_used.as_deref() {
                    for (value, &m) in workspace.psi.iter_mut().zip(w) {
                        *value *= m;
                    }
                }
                let m = workspace.psi.len() as f64;
                Ok(Some(workspace.psi.iter().sum::<f64>() / m))
            },
        )
    }
}

/// One penalized-bootstrap replicate after its indexes have been drawn.
enum PenalizedDraw {
    Done(f64, Vec<f64>),
    Failed,
    Cancelled,
}

/// Stream tag of the penalized refit bootstrap's resample plan.
const PENALIZED_BOOTSTRAP_STREAM: u64 = 0xA1D7_0001;

/// Draw replicate `replicate` into reused buffers on the penalized bootstrap stream.
///
/// Cluster labels draw whole clusters. A series block draws contiguous blocks.
/// Otherwise each row is one subject. `idx` follows the draw, so a cluster
/// replicate may not have length `n`.
#[allow(
    clippy::too_many_arguments,
    reason = "the draw is fixed by the replicate, the grouping, and the declared design together"
)]
fn fill_penalized_bootstrap_indexes(
    n: usize,
    replicate: u64,
    ctx: &ExecutionContext,
    ids: &mut Vec<antecedent_data::GlobalId>,
    idx: &mut Vec<usize>,
    clusters: Option<&[u32]>,
    series_block: Option<usize>,
    declared: crate::util::SamplingDeclaration,
) -> Result<(), EstimationError> {
    use antecedent_core::{DeclaredSamplingDesign, ResamplingRule};
    use antecedent_data::{DataError, ResamplingPlan};

    let design_err = |error: antecedent_core::SamplingDesignError| {
        EstimationError::Data(DataError::SamplingDesign(error))
    };
    let plan = if clusters.is_some() {
        // Under independent subject trajectories the labels are subject ids, and drawing
        // whole subjects is that design's own rule; every other design draws whole clusters.
        let design = declared.resolve(DeclaredSamplingDesign::ClusteredPanel);
        let rule = if matches!(design, DeclaredSamplingDesign::IndependentSubjectTrajectories) {
            ResamplingRule::RowSubjectIid
        } else {
            ResamplingRule::WholeCluster
        };
        design.apply_resampling(rule).map_err(design_err)?;
        ResamplingPlan::ClusterBootstrap { cluster: antecedent_core::VariableId::from_raw(0) }
    } else if let Some(length) = series_block {
        declared
            .resolve(DeclaredSamplingDesign::DependentTimeSeries)
            .apply_resampling(ResamplingRule::ContiguousBlock)
            .map_err(design_err)?;
        ResamplingPlan::MovingBlock { length }
    } else {
        match declared {
            crate::util::SamplingDeclaration::Undeclared => {
                DeclaredSamplingDesign::IndependentSubjectTrajectories
                    .apply_resampling(ResamplingRule::RowSubjectIid)
            }
            crate::util::SamplingDeclaration::Declared(design) => {
                design.apply_resampling(ResamplingRule::RowIid)
            }
        }
        .map_err(design_err)?;
        ResamplingPlan::IidBootstrap
    };
    antecedent_data::resample::fill_resample_index_replicate(
        plan,
        n,
        clusters,
        ctx,
        PENALIZED_BOOTSTRAP_STREAM,
        replicate,
        ids,
    )?;
    idx.clear();
    idx.try_reserve_exact(ids.len()).map_err(|e| {
        EstimationError::data_msg(format!("bootstrap index allocation failed: {e}"))
    })?;
    for id in ids.iter() {
        idx.push(usize::try_from(id.get()).expect("row id"));
    }
    Ok(())
}

/// Evaluate penalized replicates with one index buffer per worker.
///
/// Serial contexts pass `ctx` through. A parallel context passes
/// [`ExecutionContext::serial_inner`] and stops scheduling indexes above the first error,
/// matching [`ExecutionContext::map_indexed`].
fn map_penalized_replicates<F>(
    ctx: &ExecutionContext,
    n_rep: usize,
    f: F,
) -> Result<Vec<PenalizedDraw>, EstimationError>
where
    F: Fn(
            usize,
            &ExecutionContext,
            &mut Vec<antecedent_data::GlobalId>,
            &mut Vec<usize>,
        ) -> Result<PenalizedDraw, EstimationError>
        + Sync,
{
    ctx.map_indexed_with_scratch(
        n_rep,
        || (Vec::new(), Vec::new()),
        |replicate, inner, scratch| f(replicate, inner, &mut scratch.0, &mut scratch.1),
    )
}

/// The problem a penalized refit bootstrap resamples.
///
/// The fold seed and replicate nuisance are written onto a clone only when they differ
/// from `problem`. A problem that already carries both matches that clone, so it is reused.
fn penalized_bootstrap_base(
    problem: &PreparedPropensityProblem,
    fold_seed: u64,
    nuisance: PropensityNuisance,
) -> std::borrow::Cow<'_, PreparedPropensityProblem> {
    if problem.fold_seed == fold_seed && problem.propensity == nuisance {
        std::borrow::Cow::Borrowed(problem)
    } else {
        let mut seeded = problem.clone();
        seeded.fold_seed = fold_seed;
        seeded.propensity = nuisance;
        std::borrow::Cow::Owned(seeded)
    }
}

/// The prepared problem restricted to the resampled rows `idx`. Copies of one original row
/// keep that row's identity (and shared fold), so no unit is both in a training set and in
/// its own validation fold. `weights` are the custom-target weights (`None` = uniform).
pub(crate) fn resampled_problem(
    problem: &PreparedPropensityProblem,
    idx: &[usize],
    weights: Option<&[f64]>,
) -> PreparedPropensityProblem {
    let mut p = problem.clone();
    let mut design = Vec::new();
    select_rows_colmajor(
        &problem.design_matrix,
        problem.nrows,
        problem.design_ncols,
        idx,
        &mut design,
    );
    p.design_matrix = design.into();
    p.nrows = idx.len();
    p.treatment = gather(&problem.treatment, idx).into();
    p.outcome = gather(&problem.outcome, idx).into();
    p.row_index = idx.iter().map(|&i| problem.row_index[i]).collect::<Vec<_>>().into();
    p.fold_assignment = problem
        .fold_assignment
        .as_deref()
        .map(|ids| idx.iter().map(|&i| ids[i]).collect::<Vec<_>>().into());
    if let Some(w) = weights {
        p.target_weights = Some(gather(w, idx).into());
    }
    if let Some(sampling) = problem.sampling_weights.as_ref() {
        p.sampling_weights = Some(sampling.subset(idx));
    }
    p
}

/// Extract rows `idx` from a column-major `nrows × ncols` matrix into a fresh column-major
/// `idx.len() × ncols` buffer.
pub(crate) fn select_rows_colmajor(
    matrix: &[f64],
    nrows: usize,
    ncols: usize,
    idx: &[usize],
    out: &mut Vec<f64>,
) {
    crate::util::gather_cols_into(out, matrix, nrows, ncols, idx);
}

fn select_values(values: &[f64], idx: &[usize], out: &mut Vec<f64>) {
    crate::util::gather_into(out, values, idx);
}

/// Fit separate OLS outcome models on the control (`T=0`) and treated (`T=1`) arms of
/// `design_matrix` (column-major `[1 | Z…]`), returning `(beta0, beta1)`.
///
/// # Errors
///
/// Empty treated/control arm, or an OLS backend failure (e.g. rank deficiency within an arm).
#[allow(
    dead_code,
    reason = "the unweighted entry point of fit_outcome_models_weighted, kept for callers without weights"
)]
pub(crate) fn fit_outcome_models(
    design_matrix: &[f64],
    nrows: usize,
    ncols: usize,
    treatment: &[f64],
    outcome: &[f64],
    backend: FaerBackend,
    workspace: &mut AipwWorkspace,
) -> Result<(Vec<f64>, Vec<f64>), EstimationError> {
    fit_outcome_models_weighted(
        design_matrix,
        nrows,
        ncols,
        treatment,
        outcome,
        None,
        backend,
        workspace,
    )
}

/// [`fit_outcome_models`] under optional prior (sampling) weights: each arm's model is the
/// weighted least-squares fit, computed as OLS on rows scaled by the square root of their
/// weights.
///
/// # Errors
///
/// As [`fit_outcome_models`].
#[allow(
    clippy::too_many_arguments,
    reason = "the unweighted entry point's arguments plus the weights"
)]
pub(crate) fn fit_outcome_models_weighted(
    design_matrix: &[f64],
    nrows: usize,
    ncols: usize,
    treatment: &[f64],
    outcome: &[f64],
    weights: Option<&[f64]>,
    backend: FaerBackend,
    workspace: &mut AipwWorkspace,
) -> Result<(Vec<f64>, Vec<f64>), EstimationError> {
    let (treated_idx, control_idx) = split_by_treatment(treatment);
    if treated_idx.is_empty() || control_idx.is_empty() {
        return Err(EstimationError::data_msg(
            "AIPW outcome regression requires both treated and control rows",
        ));
    }

    select_rows_colmajor(design_matrix, nrows, ncols, &control_idx, &mut workspace.control_design);
    select_values(outcome, &control_idx, &mut workspace.control_outcome);
    if let Some(w) = weights {
        root_weight_rows(
            &mut workspace.control_design,
            &mut workspace.control_outcome,
            ncols,
            &control_idx,
            w,
        );
    }
    let fit0 = backend
        .least_squares(
            &workspace.control_design,
            control_idx.len(),
            ncols,
            &workspace.control_outcome,
            &mut workspace.outcome,
        )
        .map_err(crate::error::EstimationError::from)?;
    let beta0 = fit0.coefficients;

    select_rows_colmajor(design_matrix, nrows, ncols, &treated_idx, &mut workspace.treated_design);
    select_values(outcome, &treated_idx, &mut workspace.treated_outcome);
    if let Some(w) = weights {
        root_weight_rows(
            &mut workspace.treated_design,
            &mut workspace.treated_outcome,
            ncols,
            &treated_idx,
            w,
        );
    }
    let fit1 = backend
        .least_squares(
            &workspace.treated_design,
            treated_idx.len(),
            ncols,
            &workspace.treated_outcome,
            &mut workspace.outcome,
        )
        .map_err(crate::error::EstimationError::from)?;
    let beta1 = fit1.coefficients;

    Ok((beta0, beta1))
}

/// Multiply the rows of an arm's column-major design and outcome by the square root of the
/// weight of the source row each came from (`rows[r]` indexes `weights`).
fn root_weight_rows(
    design: &mut [f64],
    outcome: &mut [f64],
    ncols: usize,
    rows: &[usize],
    weights: &[f64],
) {
    let m = rows.len();
    for (r, &source) in rows.iter().enumerate() {
        let root = weights[source].sqrt();
        outcome[r] *= root;
        for c in 0..ncols {
            design[c * m + r] *= root;
        }
    }
}

/// Predict `design · coef` for every row of a column-major `nrows × ncols` design.
pub(crate) fn predict_colmajor(
    design_matrix: &[f64],
    nrows: usize,
    ncols: usize,
    coef: &[f64],
    out: &mut Vec<f64>,
) {
    out.clear();
    out.resize(nrows, 0.0);
    for (r, pred) in out.iter_mut().enumerate() {
        let mut s = 0.0;
        for c in 0..ncols {
            s += design_matrix[c * nrows + r] * coef[c];
        }
        *pred = s;
    }
}

/// Compute AIPW per-unit IF values for ATE / ATT / ATC.
#[cfg(test)]
fn aipw_psi(
    treatment: &[f64],
    outcome: &[f64],
    propensity: &[f64],
    mu0: &[f64],
    mu1: &[f64],
    target: &TargetPopulation,
    out: &mut Vec<f64>,
) -> Result<(), EstimationError> {
    aipw_psi_weighted(treatment, outcome, propensity, mu0, mu1, target, None, out)
}

/// [`aipw_psi`] with optional sampling weights (mean one): the ATT / ATC arm share `π̂` is
/// the weighted share, so `mean(m_i ψ_i)` is the weighted-population effect. The ATE score
/// per row does not depend on the weights.
#[allow(
    clippy::too_many_arguments,
    reason = "the score reads the six aligned nuisance and data columns, the target and the weights"
)]
fn aipw_psi_weighted(
    treatment: &[f64],
    outcome: &[f64],
    propensity: &[f64],
    mu0: &[f64],
    mu1: &[f64],
    target: &TargetPopulation,
    weights: Option<&[f64]>,
    out: &mut Vec<f64>,
) -> Result<(), EstimationError> {
    let weight = |i: usize| weights.map_or(1.0, |w| w[i]);
    let n_rows = treatment.len() as f64;
    out.clear();
    out.reserve(treatment.len());
    match target {
        TargetPopulation::AllObserved | TargetPopulation::Predicate(_) => {
            for (((&t, &y), &e), (&m0, &m1)) in
                treatment.iter().zip(outcome).zip(propensity).zip(mu0.iter().zip(mu1))
            {
                let augmented = (m1 - m0) + (t / e) * (y - m1) - ((1.0 - t) / (1.0 - e)) * (y - m0);
                out.push(augmented);
            }
        }
        TargetPopulation::Treated => {
            let pi = treatment
                .iter()
                .enumerate()
                .filter(|(_, t)| **t > 0.5)
                .map(|(i, _)| weight(i))
                .sum::<f64>()
                / n_rows;
            if pi <= 0.0 {
                return Err(EstimationError::data_msg("ATT requires treated units"));
            }
            for (((&t, &y), &e), (&m0, &m1)) in
                treatment.iter().zip(outcome).zip(propensity).zip(mu0.iter().zip(mu1))
            {
                let aug = (t / pi) * (m1 - m0) + (t / pi) * (y - m1)
                    - ((1.0 - t) / pi) * (e / (1.0 - e)) * (y - m0);
                out.push(aug);
            }
        }
        TargetPopulation::Untreated => {
            let pi0 = treatment
                .iter()
                .enumerate()
                .filter(|(_, t)| **t <= 0.5)
                .map(|(i, _)| weight(i))
                .sum::<f64>()
                / n_rows;
            if pi0 <= 0.0 {
                return Err(EstimationError::data_msg("ATC requires control units"));
            }
            for (((&t, &y), &e), (&m0, &m1)) in
                treatment.iter().zip(outcome).zip(propensity).zip(mu0.iter().zip(mu1))
            {
                let aug = ((1.0 - t) / pi0) * (m1 - m0) + (t / pi0) * ((1.0 - e) / e) * (y - m1)
                    - ((1.0 - t) / pi0) * (y - m0);
                out.push(aug);
            }
        }
        _ => {
            return Err(EstimationError::unsupported(
                "AIPW unsupported target population for IF construction",
            ));
        }
    }
    Ok(())
}

/// Turn ATT/ATC plug-in terms into influence functions.
///
/// `aipw_psi` returns `N_i / π̂` for ATT (`N_i = T(Y − μ₀) − (1−T)·e/(1−e)·(Y − μ₀)`)
/// and the mirror image over `1 − π̂` for ATC, whose mean is the estimate. The
/// estimate divides by the *estimated* arm share, so its influence function is
/// `(N_i − τ·T_i) / π` (ATC: `(N_i − τ·(1−T_i)) / π₀`), which is the efficient
/// DR influence function and needs no propensity-score projection when both
/// nuisance models are consistent (with one misspecified it omits the
/// nuisance-estimation terms, so the analytic SE is not doubly robust). Using
/// the plug-in terms as the IF (dropping `−τ·T_i/π`) and then
/// projected off the logistic scores; at nominal 0.95 that measured 0.980 ATT /
/// 0.863 ATC coverage (`calibration_coverage::aipw_at{t,c}_hc1_ci_coverage`).
/// ATE / predicate targets already average over every row and are unchanged.
fn center_population_psi(
    psi: &mut [f64],
    treatment: &[f64],
    target: &TargetPopulation,
    estimate: f64,
    weights: Option<&[f64]>,
) {
    let n = treatment.len() as f64;
    let weight = |i: usize| weights.map_or(1.0, |w| w[i]);
    let arm_indicator =
        |t: f64, treated: bool| -> f64 { if (t > 0.5) == treated { 1.0 } else { 0.0 } };
    let treated = match target {
        TargetPopulation::Treated => true,
        TargetPopulation::Untreated => false,
        _ => return,
    };
    let share = treatment
        .iter()
        .enumerate()
        .map(|(i, &t)| weight(i) * arm_indicator(t, treated))
        .sum::<f64>()
        / n;
    if share <= 0.0 {
        return;
    }
    for (i, (value, &t)) in psi.iter_mut().zip(treatment).enumerate() {
        *value -= estimate * weight(i) * arm_indicator(t, treated) / share;
    }
}

/// Add the first-order terms for the *estimated* nuisances to the ATE score ψ, so that
/// `se_analytic` is not conditional on ê, μ̂₀, μ̂₁ as known.
///
/// The estimator solves a stacked system: the logistic score `x(T−e)`, and per-arm OLS scores
/// `T·x(Y−μ₁)`, `(1−T)·x(Y−μ₀)`. Each block contributes `Bᵀ I⁻¹ s_i` with `B` the mean
/// derivative of ψ in that block's parameters (see [`crate::se::add_nuisance_correction`]):
///
/// ```text
/// ∂ψ/∂γ  = −[T(Y−μ₁)(1−e)/e + (1−T)(Y−μ₀)e/(1−e)]·x      (0 where ê was clipped)
/// ∂ψ/∂β₁ =  (1 − T/e)·x        ∂ψ/∂β₀ = −(1 − (1−T)/(1−e))·x
/// ```
///
/// This is exact whether or not the propensity model is correct. The projection of ψ on the
/// logistic score, used previously, equals the propensity term only under the information
/// equality and, under a misspecified propensity with heterogeneous effects, removes real
/// variance; the outcome-model terms vanish only when the propensity is correct. The logistic
/// score uses the raw fitted `ê` (it sums to zero at the MLE), the derivative the clipped one.
/// A singular block information matrix refuses rather than publishing an uncorrected SE.
fn correct_aipw_psi_for_nuisances(
    psi: &mut [f64],
    fit: &AipwNuisanceFit<'_>,
) -> Result<(), EstimationError> {
    let n = psi.len();
    if fit.ncols == 0 || n < 2 {
        return Ok(());
    }
    let mut e_score = vec![0.0; n];
    let mut e_info = vec![0.0; n];
    let mut e_deriv = vec![0.0; n];
    let mut b1_score = vec![0.0; n];
    let mut b1_info = vec![0.0; n];
    let mut b1_deriv = vec![0.0; n];
    let mut b0_score = vec![0.0; n];
    let mut b0_info = vec![0.0; n];
    let mut b0_deriv = vec![0.0; n];
    for i in 0..n {
        let (t, y, e, raw) = (fit.treatment[i], fit.outcome[i], fit.e_used[i], fit.e_raw[i]);
        let (r1, r0) = (y - fit.mu1[i], y - fit.mu0[i]);
        e_score[i] = t - raw;
        e_info[i] = raw * (1.0 - raw);
        // Clipped rows do not depend on γ locally.
        #[allow(
            clippy::float_cmp,
            reason = "e is either exactly the raw propensity or the clipped bound, so exact inequality detects a clipped row"
        )]
        let clipped = e != raw;
        e_deriv[i] =
            if clipped { 0.0 } else { -(t * r1 * (1.0 - e) / e + (1.0 - t) * r0 * e / (1.0 - e)) };
        b1_score[i] = t * r1;
        b1_info[i] = t;
        b1_deriv[i] = 1.0 - t / e;
        b0_score[i] = (1.0 - t) * r0;
        b0_info[i] = 1.0 - t;
        b0_deriv[i] = -(1.0 - (1.0 - t) / (1.0 - e));
    }
    if let Some(weights) = fit.weights {
        // Weighted stacked system: every block's score, information and derivative carries
        // the row's sampling weight.
        for block in [
            &mut e_score,
            &mut e_info,
            &mut e_deriv,
            &mut b1_score,
            &mut b1_info,
            &mut b1_deriv,
            &mut b0_score,
            &mut b0_info,
            &mut b0_deriv,
        ] {
            for (value, &m) in block.iter_mut().zip(weights) {
                *value *= m;
            }
        }
    }
    let singular = "singular AIPW nuisance-score information; refusing an uncorrected analytic SE";
    // Each block's derivative is a function of the data only, so the terms add independently.
    for (score, info, deriv) in [
        (&e_score, &e_info, &e_deriv),
        (&b1_score, &b1_info, &b1_deriv),
        (&b0_score, &b0_info, &b0_deriv),
    ] {
        crate::se::add_nuisance_correction(
            psi, fit.design, fit.ncols, score, info, deriv, singular,
        )?;
    }
    Ok(())
}

/// Rows and fitted nuisances behind an AIPW ATE score (aligned, length = retained rows).
struct AipwNuisanceFit<'a> {
    treatment: &'a [f64],
    outcome: &'a [f64],
    /// Clipped propensity used in ψ.
    e_used: &'a [f64],
    /// Raw fitted propensity (the logistic MLE's own scores).
    e_raw: &'a [f64],
    mu0: &'a [f64],
    mu1: &'a [f64],
    design: &'a [f64],
    ncols: usize,
    /// Sampling weights of the rows (mean one), when declared.
    weights: Option<&'a [f64]>,
}

/// Overlap report of the treated arm's cross-fitted propensities, when the table carries one.
fn treated_arm_overlap_report(
    table: &crate::scores::ScoreTable,
    problem: &PreparedPropensityProblem,
) -> Option<OverlapReport> {
    let e_hat = table
        .columns
        .iter()
        .position(|c| c.arm == 1 && c.threshold.is_none())
        .and_then(|col| table.propensities.get(col * table.n_rows..(col + 1) * table.n_rows))?;
    Some(crate::propensity::propensity_overlap_report(
        problem,
        e_hat,
        None,
        IpwTarget::from_population(&problem.target_population).ok(),
    ))
}

#[cfg(test)]
#[allow(
    clippy::many_single_char_names,
    clippy::float_cmp,
    reason = "this unit-test module compares floats that are copied, clamped or hand-set without rounding, so exact equality is intended"
)]
mod tests {
    use antecedent_core::StreamDomain;

    use std::sync::Arc;

    use antecedent_core::{
        CausalSchemaBuilder, ExecutionContext, MeasurementSpec, RoleHint, SmallRoleSet,
        TargetPopulation, ValueType, VariableId,
    };
    use antecedent_data::{
        Float64Column, OwnedColumn, OwnedColumnarStorage, TabularData, ValidityBitmap,
    };
    use antecedent_expr::ExprId;
    use antecedent_expr::IdentifiedEstimand;
    use antecedent_identify::{
        DerivationTrace, IdentificationPerformanceRecord, IdentificationResult,
    };

    use super::*;
    use crate::overlap::OverlapPolicy;
    use antecedent_kernels::standard_normal;

    /// Confounded SCM: `Z ~ N(0,1)`, `T ~ Bernoulli(logit(-0.5 + Z))`, `Y = 2T + Z + noise`.
    /// True ATE = 2. Matches the propensity-estimator test fixture (`crate::propensity`).
    fn confounded_scm(n: usize, seed: u64) -> (TabularData, IdentifiedEstimand) {
        let (t, y, z) = confounded_columns(n, seed);
        build_dataset(t, y, z)
    }

    fn confounded_columns(n: usize, seed: u64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut rng =
            ExecutionContext::for_tests(seed).rng.stream_for(StreamDomain::Estimate, 0x1234_u64);

        let mut z = vec![0.0; n];
        let mut t = vec![0.0; n];
        let mut y = vec![0.0; n];
        for i in 0..n {
            let zi = standard_normal(&mut rng);
            let logit = -0.5 + zi;
            let p = antecedent_kernels::sigmoid(logit);
            let ti = if rng.next_f64() < p { 1.0 } else { 0.0 };
            let noise = standard_normal(&mut rng) * 0.5;
            z[i] = zi;
            t[i] = ti;
            y[i] = 2.0 * ti + zi + noise;
        }
        (t, y, z)
    }

    fn build_dataset(t: Vec<f64>, y: Vec<f64>, z: Vec<f64>) -> (TabularData, IdentifiedEstimand) {
        let n = t.len();
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

    fn identified_aipw(data_estimand: &IdentifiedEstimand) -> IdentificationResult {
        let query = antecedent_core::AverageEffectQuery::binary_ate(
            VariableId::from_raw(0),
            VariableId::from_raw(1),
        );
        let mut arena = antecedent_expr::CausalExprArena::new();
        let functional = arena.backdoor_ate(
            query.treatment,
            query.outcome,
            &[VariableId::from_raw(2)],
            antecedent_core::Value::f64(1.0),
            antecedent_core::Value::f64(0.0),
        );
        IdentificationResult::identified(
            antecedent_core::CausalQuery::average_effect(query),
            vec![IdentifiedEstimand::backdoor(
                "backdoor.adjustment",
                Arc::clone(&data_estimand.adjustment_set),
                functional,
            )],
            arena,
            DerivationTrace::default(),
            AssumptionSet::new(),
            IdentificationPerformanceRecord::default(),
        )
    }

    #[test]
    fn aipw_runs_the_complete_interval_procedure_on_both_bootstrap_routes() {
        use crate::util::interval_test_support::{assert_procedure, met_early, never_met};
        let (data, estimand) = confounded_scm(300, 12);
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let (cap, wave) = (90, 30);
        let est = AipwAte { bootstrap_replicates: cap, ..AipwAte::new() };
        let fit = |est: &AipwAte| {
            let prep = est.prepare(&data, &estimand, &query).unwrap();
            est.fit(
                &prep,
                &mut AipwWorkspace::default(),
                &ExecutionContext::for_tests(3),
                AssumptionSet::new(),
            )
            .unwrap()
        };
        // The cross-fitted route (untrimmed AllObserved).
        assert_procedure(
            &fit(&est),
            &fit(&est.clone().with_bootstrap_interval(never_met(wave))),
            &fit(&est.clone().with_bootstrap_interval(met_early(wave, 60))),
            cap,
            wave,
        );
        // The penalized refit bootstrap keeps its penalties beside the same procedure.
        let ridge = AipwAte { bootstrap_replicates: cap, ..ridge_estimator(&[0.5, 5.0], 3) };
        let fixed = fit(&ridge);
        let never = fit(&ridge.clone().with_bootstrap_interval(never_met(wave)));
        let early = fit(&ridge.clone().with_bootstrap_interval(met_early(wave, 60)));
        assert_procedure(&fixed, &never, &early, cap, wave);
        let kept = |e: &EffectEstimate| {
            e.diagnostics
                .penalized
                .as_ref()
                .unwrap()
                .variance
                .as_ref()
                .unwrap()
                .replicate_penalties
                .len()
        };
        assert_eq!(kept(&never), kept(&fixed));
        assert_eq!(
            kept(&early),
            usize::try_from(early.uncertainty.bootstrap.replicates_ok.unwrap()).unwrap()
        );
    }

    #[test]
    fn checked_aipw_executes_retained_target_and_rebinds_rows() {
        let (data, estimand) = confounded_scm(2400, 9182);
        let identification = identified_aipw(&estimand);
        let estimator = AipwAte { bootstrap_replicates: 0, ..AipwAte::new() };
        let checked = estimator.prepare_checked(&data, &identification, 0).unwrap();
        assert_eq!(checked.lowering().functional, checked.target().functional);
        assert_eq!(checked.lowering().treatment, VariableId::from_raw(0));
        assert_eq!(checked.lowering().outcome, VariableId::from_raw(1));
        assert_eq!(checked.lowering().adjustment.as_ref(), &[VariableId::from_raw(2)]);
        assert_eq!(checked.lowering().procedure, CheckedAipwProcedure::CrossFittedLogisticOls);
        assert_eq!(checked.lowering().rows.as_ref(), checked.problem().row_index.as_ref());
        let rebound = estimator.rebind_checked(&checked, &data).unwrap();
        assert_eq!(rebound.program().mapping(), checked.program().mapping());
        assert_eq!(rebound.lowering().rows.as_ref(), checked.lowering().rows.as_ref());

        let mut workspace = AipwWorkspace::default();
        let estimate = estimator
            .fit_checked(&checked, &mut workspace, &ExecutionContext::for_tests(441))
            .unwrap();
        assert!((estimate.ate - 2.0).abs() < 0.15, "AIPW estimate {}", estimate.ate);
    }

    #[test]
    fn checked_aipw_refuses_target_with_wrong_roles() {
        let (data, estimand) = confounded_scm(300, 33);
        let mut identification = identified_aipw(&estimand);
        identification.estimands[0] = IdentifiedEstimand::backdoor(
            "backdoor.adjustment",
            Arc::from([VariableId::from_raw(0)]),
            identification.estimands[0].functional,
        );
        let estimator = AipwAte { bootstrap_replicates: 0, ..AipwAte::new() };
        assert!(estimator.prepare_checked(&data, &identification, 0).is_err());
    }

    /// A trimming policy lowers to the full-sample common-support procedure and
    /// the checked fit is the estimator's own trimmed fit, bit for bit.
    #[test]
    fn checked_trimmed_aipw_retains_its_policy_and_matches_the_plain_fit() {
        let (data, estimand) = confounded_scm(1200, 4171);
        let identification = identified_aipw(&estimand);
        let overlap = OverlapPolicy::RequireDiagnostics { clip: Some(0.01), trim: Some(0.05) };
        let estimator = AipwAte { bootstrap_replicates: 6, overlap, ..AipwAte::new() };
        let checked = estimator.prepare_checked(&data, &identification, 0).unwrap();
        assert_eq!(checked.lowering().procedure, CheckedAipwProcedure::TrimmedLogisticOls);
        assert_eq!(checked.lowering().overlap, overlap);
        assert_eq!(checked.lowering().folds, 0);
        assert_eq!(checked.problem().overlap, overlap);
        let rebound = estimator.rebind_checked(&checked, &data).unwrap();
        assert_eq!(rebound.lowering().procedure, CheckedAipwProcedure::TrimmedLogisticOls);

        let mut workspace = AipwWorkspace::default();
        let sealed = estimator
            .fit_checked(&checked, &mut workspace, &ExecutionContext::for_tests(441))
            .unwrap();
        let query = identification.average_effect().unwrap();
        let problem = estimator.prepare(&data, &identification.estimands[0], query).unwrap();
        let plain = estimator
            .fit(&problem, &mut workspace, &ExecutionContext::for_tests(441), AssumptionSet::new())
            .unwrap();
        assert_eq!(sealed.ate.to_bits(), plain.ate.to_bits());
        assert_eq!(sealed.se_analytic().to_bits(), plain.se_analytic().to_bits());
        assert_eq!(sealed.se_bootstrap().map(f64::to_bits), plain.se_bootstrap().map(f64::to_bits));
        assert!(sealed.scores.table.is_none(), "the trimmed fit has no cross-fitted score table");
        assert!(
            sealed
                .diagnostics
                .overlap_report
                .as_ref()
                .is_some_and(|report| report.excluded_fraction > 0.0)
        );

        // An estimator whose policy differs from the receipt cannot execute it.
        let untrimmed = AipwAte { bootstrap_replicates: 6, ..AipwAte::new() };
        assert!(untrimmed.fit_checked(&checked, &mut workspace, &ctx()).is_err());
        assert!(untrimmed.rebind_checked(&checked, &data).is_err());
    }

    fn ctx() -> ExecutionContext {
        ExecutionContext::for_tests(7)
    }

    /// The facade attaches the bootstrap to a replicate-free point fit; every AIPW route
    /// (cross-fitted untrimmed ATE, trimmed/ATT nuisance refit) must publish the same estimate.
    #[test]
    fn aipw_attached_bootstrap_equals_the_one_shot_fit_on_every_route() {
        let (data, estimand) = confounded_scm(400, 3);
        let ate = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let att = ate.clone().with_target_population(TargetPopulation::Treated);
        for query in [ate, att] {
            let boot = AipwAte { bootstrap_replicates: 30, ..AipwAte::new() };
            let prep = boot.prepare(&data, &estimand, &query).unwrap();
            let mut ws = AipwWorkspace::default();
            let full = boot.fit(&prep, &mut ws, &ctx(), AssumptionSet::new()).unwrap();
            let point_only = AipwAte { bootstrap_replicates: 0, ..boot.clone() };
            let point = point_only.fit(&prep, &mut ws, &ctx(), AssumptionSet::new()).unwrap();
            assert!(point.se_bootstrap().is_none());
            let attached = boot.attach_bootstrap(&prep, &mut ws, &ctx(), point).unwrap();
            assert_eq!(attached.ate.to_bits(), full.ate.to_bits());
            assert!(full.se_bootstrap().is_some());
            assert_eq!(attached.se_bootstrap(), full.se_bootstrap());
            assert_eq!(
                attached.uncertainty.bootstrap.replicates_ok,
                full.uncertainty.bootstrap.replicates_ok
            );
        }
    }

    #[test]
    fn aipw_recovers_ate_two() {
        let (data, estimand) = confounded_scm(800, 1);
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let est = AipwAte { bootstrap_replicates: 30, ..AipwAte::new() };
        let prep = est.prepare(&data, &estimand, &query).unwrap();
        let mut ws = AipwWorkspace::default();
        let effect = est.fit(&prep, &mut ws, &ctx(), AssumptionSet::new()).unwrap();
        assert!((effect.ate - 2.0).abs() < 0.3, "ate={}", effect.ate);
        assert!(effect.se_bootstrap().is_some());
        assert!(effect.diagnostics.overlap_report.is_some());
    }

    #[test]
    fn aipw_rejects_explicit_override() {
        let (data, estimand) = confounded_scm(200, 2);
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let est = AipwAte { overlap: OverlapPolicy::ExplicitOverride, ..AipwAte::new() };
        let err = est.prepare(&data, &estimand, &query).unwrap_err();
        assert!(matches!(err, EstimationError::Overlap { .. }));
    }

    #[test]
    fn aipw_recovers_att_two() {
        let (data, estimand) = confounded_scm(1_200, 3);
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1))
                .with_target_population(TargetPopulation::Treated);
        let est = AipwAte { bootstrap_replicates: 0, ..AipwAte::new() };
        let prep = est.prepare(&data, &estimand, &query).unwrap();
        let mut ws = AipwWorkspace::default();
        let fit = est.fit(&prep, &mut ws, &ctx(), AssumptionSet::new()).unwrap();
        assert!((fit.ate - 2.0).abs() < 0.35, "att={}", fit.ate);
    }

    /// ATT IF remains unbiased when μ₁ is misspecified but propensity is correct.
    /// The old formula divided the treated residual by `e`, breaking double robustness.
    #[test]
    fn att_if_doubly_robust_under_mu1_misspecification() {
        let n = 4_000usize;
        let mut rng =
            ExecutionContext::for_tests(42).rng.stream_for(StreamDomain::Estimate, 0xA11u64);
        let mut t = vec![0.0; n];
        let mut y = vec![0.0; n];
        let mut e = vec![0.0; n];
        let mut mu0 = vec![0.0; n];
        let mut mu1 = vec![0.0; n];
        for i in 0..n {
            let z = standard_normal(&mut rng);
            let logit = -0.5 + z;
            let pi_i = antecedent_kernels::sigmoid(logit);
            let ti = if rng.next_f64() < pi_i { 1.0 } else { 0.0 };
            let noise = standard_normal(&mut rng) * 0.5;
            t[i] = ti;
            y[i] = 2.0 * ti + z + noise;
            e[i] = pi_i;
            // Correct μ₀; deliberately wrong μ₁ (constant 0 instead of 2+z).
            mu0[i] = z;
            mu1[i] = 0.0;
        }
        let mut psi = Vec::new();
        aipw_psi(&t, &y, &e, &mu0, &mu1, &TargetPopulation::Treated, &mut psi).unwrap();
        let att = psi.iter().sum::<f64>() / psi.len() as f64;
        assert!(
            (att - 2.0).abs() < 0.15,
            "ATT IF mean under μ₁ misspecification should stay near 2; got {att}"
        );
    }

    /// Guide MATH-003 deterministic counterexample: correct e, wrong m0 → ATC = 2.
    #[test]
    fn atc_if_deterministic_counterexample() {
        // P(T)=0.5, e=0.5, Y(0)=1, Y(1)=3, m1=3, m0=0 → true ATC = 2.
        let t = vec![1.0, 1.0, 0.0, 0.0];
        let y = vec![3.0, 3.0, 1.0, 1.0];
        let e = vec![0.5, 0.5, 0.5, 0.5];
        let mu0 = vec![0.0, 0.0, 0.0, 0.0];
        let mu1 = vec![3.0, 3.0, 3.0, 3.0];
        let mut psi = Vec::new();
        aipw_psi(&t, &y, &e, &mu0, &mu1, &TargetPopulation::Untreated, &mut psi).unwrap();
        let atc = psi.iter().sum::<f64>() / psi.len() as f64;
        assert!((atc - 2.0).abs() < 1e-12, "deterministic ATC IF mean should be 2; got {atc}");
    }

    /// ATC IF remains unbiased when μ₀ is misspecified but propensity is correct.
    #[test]
    fn atc_if_doubly_robust_under_mu0_misspecification() {
        let n = 4_000usize;
        let mut rng =
            ExecutionContext::for_tests(42).rng.stream_for(StreamDomain::Estimate, 0xA7Cu64);
        let mut t = vec![0.0; n];
        let mut y = vec![0.0; n];
        let mut e = vec![0.0; n];
        let mut mu0 = vec![0.0; n];
        let mut mu1 = vec![0.0; n];
        for i in 0..n {
            let z = standard_normal(&mut rng);
            let logit = -0.5 + z;
            let pi_i = antecedent_kernels::sigmoid(logit);
            let ti = if rng.next_f64() < pi_i { 1.0 } else { 0.0 };
            let noise = standard_normal(&mut rng) * 0.5;
            t[i] = ti;
            y[i] = 2.0 * ti + z + noise;
            e[i] = pi_i;
            // Correct μ₁; deliberately wrong μ₀ (constant 0 instead of z).
            mu0[i] = 0.0;
            mu1[i] = 2.0 + z;
        }
        let mut psi = Vec::new();
        aipw_psi(&t, &y, &e, &mu0, &mu1, &TargetPopulation::Untreated, &mut psi).unwrap();
        let atc = psi.iter().sum::<f64>() / psi.len() as f64;
        assert!(
            (atc - 2.0).abs() < 0.15,
            "ATC IF mean under μ₀ misspecification should stay near 2; got {atc}"
        );
    }

    /// ATC IF remains unbiased when propensity is misspecified but outcomes are correct.
    #[test]
    fn atc_if_doubly_robust_under_propensity_misspecification() {
        let n = 4_000usize;
        let mut rng =
            ExecutionContext::for_tests(43).rng.stream_for(StreamDomain::Estimate, 0xBEEFu64);
        let mut t = vec![0.0; n];
        let mut y = vec![0.0; n];
        let mut e = vec![0.0; n];
        let mut mu0 = vec![0.0; n];
        let mut mu1 = vec![0.0; n];
        for i in 0..n {
            let z = standard_normal(&mut rng);
            let logit = -0.5 + z;
            let pi_i = antecedent_kernels::sigmoid(logit);
            let ti = if rng.next_f64() < pi_i { 1.0 } else { 0.0 };
            let noise = standard_normal(&mut rng) * 0.5;
            t[i] = ti;
            y[i] = 2.0 * ti + z + noise;
            // Deliberately wrong propensity (constant 0.5).
            e[i] = 0.5;
            mu0[i] = z;
            mu1[i] = 2.0 + z;
        }
        let mut psi = Vec::new();
        aipw_psi(&t, &y, &e, &mu0, &mu1, &TargetPopulation::Untreated, &mut psi).unwrap();
        let atc = psi.iter().sum::<f64>() / psi.len() as f64;
        assert!(
            (atc - 2.0).abs() < 0.15,
            "ATC IF mean under propensity misspecification should stay near 2; got {atc}"
        );
    }

    /// ATC IF unbiased when both propensity and outcome models are correct.
    #[test]
    fn atc_if_doubly_robust_when_both_correct() {
        let n = 4_000usize;
        let mut rng =
            ExecutionContext::for_tests(44).rng.stream_for(StreamDomain::Estimate, 0xCAFEu64);
        let mut t = vec![0.0; n];
        let mut y = vec![0.0; n];
        let mut e = vec![0.0; n];
        let mut mu0 = vec![0.0; n];
        let mut mu1 = vec![0.0; n];
        for i in 0..n {
            let z = standard_normal(&mut rng);
            let logit = -0.5 + z;
            let pi_i = antecedent_kernels::sigmoid(logit);
            let ti = if rng.next_f64() < pi_i { 1.0 } else { 0.0 };
            let noise = standard_normal(&mut rng) * 0.5;
            t[i] = ti;
            y[i] = 2.0 * ti + z + noise;
            e[i] = pi_i;
            mu0[i] = z;
            mu1[i] = 2.0 + z;
        }
        let mut psi = Vec::new();
        aipw_psi(&t, &y, &e, &mu0, &mu1, &TargetPopulation::Untreated, &mut psi).unwrap();
        let atc = psi.iter().sum::<f64>() / psi.len() as f64;
        assert!(
            (atc - 2.0).abs() < 0.15,
            "ATC IF mean with both models correct should stay near 2; got {atc}"
        );
    }

    #[test]
    fn aipw_hc1_multiway_newey_west_finite_se() {
        let (data, estimand) = confounded_scm(400, 8);
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let n = 400;
        let dim_a: Vec<u32> = (0..n).map(|i| u32::try_from(i % 20).unwrap_or(0)).collect();
        let dim_b: Vec<u32> = (0..n).map(|i| u32::try_from(i % 15).unwrap_or(0)).collect();
        for kind in
            [AnalyticSeKind::Hc1, AnalyticSeKind::Multiway, AnalyticSeKind::NeweyWest { lag: 2 }]
        {
            let est = AipwAte {
                bootstrap_replicates: 0,
                se_kind: kind,
                multiway_ids: Some(vec![dim_a.clone(), dim_b.clone()]),
                ..AipwAte::new()
            };
            let prep = est.prepare(&data, &estimand, &query).unwrap();
            let mut ws = AipwWorkspace::default();
            let fit = est.fit(&prep, &mut ws, &ctx(), AssumptionSet::new()).unwrap();
            assert!(fit.se_analytic().is_finite() && fit.se_analytic() > 0.0, "kind={kind:?}");
        }
    }

    #[test]
    fn aipw_trim_excludes_extreme_propensity_unit() {
        // One treated outlier with z = -8 (raw propensity ≈ 2e-4) and y = 1000: its clipped
        // T/e term is ~100 · (1000 − μ1), which wrecks the untrimmed ψ average. Trimming on
        // the raw score must exclude it.
        let (mut t, mut y, mut z) = confounded_columns(800, 5);
        t.push(1.0);
        y.push(1000.0);
        z.push(-8.0);
        let (data, estimand) = build_dataset(t, y, z);
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));

        let untrimmed = AipwAte { bootstrap_replicates: 0, ..AipwAte::new() };
        let trimmed = AipwAte {
            overlap: OverlapPolicy::RequireDiagnostics { clip: Some(0.01), trim: Some(0.02) },
            ..untrimmed.clone()
        };
        let mut ws = AipwWorkspace::default();
        let prep = untrimmed.prepare(&data, &estimand, &query).unwrap();
        let raw = untrimmed.fit(&prep, &mut ws, &ctx(), AssumptionSet::new()).unwrap();
        let prep = trimmed.prepare(&data, &estimand, &query).unwrap();
        let clean = trimmed.fit(&prep, &mut ws, &ctx(), AssumptionSet::new()).unwrap();

        assert!((raw.ate - 2.0).abs() > 1.0, "outlier should distort untrimmed ate={}", raw.ate);
        assert!((clean.ate - 2.0).abs() < 0.35, "trimmed ate={}", clean.ate);
        let report = clean.diagnostics.overlap_report.as_ref().unwrap();
        assert!(report.excluded_fraction > 0.0, "trim must report exclusions");
    }

    #[test]
    fn aipw_works_with_efficient_backdoor_estimand() {
        let (data, mut estimand) = confounded_scm(800, 4);
        estimand.method = Arc::from("backdoor.efficient");
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let est = AipwAte { bootstrap_replicates: 0, ..AipwAte::new() };
        let prep = est.prepare(&data, &estimand, &query).unwrap();
        let mut ws = AipwWorkspace::default();
        let effect = est.fit(&prep, &mut ws, &ctx(), AssumptionSet::new()).unwrap();
        assert!((effect.ate - 2.0).abs() < 0.3, "ate={}", effect.ate);
    }

    /// Leave-one-out jackknife SE of the whole estimator (logistic + both OLS refit each time):
    /// an SE that needs no influence-function algebra, so it is an independent yardstick.
    fn jackknife_se(t: &[f64], y: &[f64], z: &[f64], est: &AipwAte) -> f64 {
        let n = t.len();
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let mut ws = AipwWorkspace::default();
        let mut loo = Vec::with_capacity(n);
        for drop in 0..n {
            let keep = |v: &[f64]| -> Vec<f64> {
                v.iter().enumerate().filter(|(i, _)| *i != drop).map(|(_, x)| *x).collect()
            };
            let (data, estimand) = build_dataset(keep(t), keep(y), keep(z));
            let prep = est.prepare(&data, &estimand, &query).unwrap();
            loo.push(est.fit(&prep, &mut ws, &ctx(), AssumptionSet::new()).unwrap().ate);
        }
        let mean = loo.iter().sum::<f64>() / n as f64;
        let ss: f64 = loo.iter().map(|v| (v - mean).powi(2)).sum();
        ((n as f64 - 1.0) / n as f64 * ss).sqrt()
    }

    #[test]
    fn ate_analytic_se_matches_jackknife_under_misspecified_propensity() {
        // True propensity depends on z²; the fitted logistic is linear in z (misspecified).
        // Outcome per arm is linear in z (correct) with CATE 2 + 1.5 z, so the estimator is
        // consistent for ATE = 2, E[∂ψ/∂γ] = 0, but E[(τ(Z)−τ)·Z(e₀−e*)] ≠ 0: projecting ψ
        // off the logistic score would delete real variance. The stacked linearisation must
        // agree with the delete-one jackknife of the full estimator.
        let n = 300usize;
        let mut rng = ExecutionContext::for_tests(61).rng.stream_for(StreamDomain::Estimate, 0x77);
        let (mut t, mut y, mut z) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        for i in 0..n {
            let zi = standard_normal(&mut rng);
            let logit = -0.5 + 0.3 * zi + 1.2 * (zi * zi - 1.0);
            let p = antecedent_kernels::sigmoid(logit);
            let ti = if rng.next_f64() < p { 1.0 } else { 0.0 };
            z[i] = zi;
            t[i] = ti;
            y[i] = 2.0 * ti + zi + 1.5 * ti * zi + 0.5 * standard_normal(&mut rng);
        }
        // A trim below every score keeps all rows but routes through the full-sample path.
        let est = AipwAte {
            bootstrap_replicates: 0,
            overlap: OverlapPolicy::RequireDiagnostics { clip: Some(0.01), trim: Some(1e-9) },
            ..AipwAte::new()
        };
        let (data, estimand) = build_dataset(t.clone(), y.clone(), z.clone());
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let prep = est.prepare(&data, &estimand, &query).unwrap();
        let fit = est.fit(&prep, &mut AipwWorkspace::default(), &ctx(), AssumptionSet::new());
        let fit = fit.unwrap();
        let jk = jackknife_se(&t, &y, &z, &est);
        let rel = (fit.se_analytic() - jk).abs() / jk;
        assert!(rel < 0.15, "analytic {} vs jackknife {jk}", fit.se_analytic());
    }

    #[test]
    fn point_estimate_does_not_depend_on_the_requested_se_kind() {
        let (data, estimand) = confounded_scm(400, 12);
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let mut ws = AipwWorkspace::default();
        let base = AipwAte { bootstrap_replicates: 0, ..AipwAte::new() };
        let prep = base.prepare(&data, &estimand, &query).unwrap();
        let iid = base.fit(&prep, &mut ws, &ctx(), AssumptionSet::new()).unwrap();
        let clusters: Vec<u32> = (0..400).map(|i| u32::try_from(i / 4).unwrap_or(0)).collect();
        for est in [
            AipwAte { se_kind: AnalyticSeKind::Hc1, ..base.clone() },
            AipwAte {
                se_kind: AnalyticSeKind::Cluster,
                cluster_ids: Some(clusters.clone()),
                ..base.clone()
            },
        ] {
            let prep = est.prepare(&data, &estimand, &query).unwrap();
            let fit = est.fit(&prep, &mut ws, &ctx(), AssumptionSet::new()).unwrap();
            assert_eq!(fit.ate.to_bits(), iid.ate.to_bits(), "{:?}", est.se_kind);
            assert!(fit.scores.table.is_some());
            assert!(fit.joint.covariance.is_none() && fit.scores.inference.is_none());
        }
    }

    fn ridge_estimator(grid: &[f64], inner_folds: usize) -> AipwAte {
        AipwAte {
            bootstrap_replicates: 0,
            propensity: PropensityNuisance::ridge_logistic(
                crate::propensity::RidgeTuning::new(grid, inner_folds).unwrap(),
            ),
            ..AipwAte::new()
        }
    }

    /// A resample that lists every row once, in order, is the original problem: the same score
    /// table and the same selections, to the bit.
    #[test]
    fn the_identity_resample_is_the_original_problem() {
        let (data, estimand) = confounded_scm(300, 71);
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let est = ridge_estimator(&[0.5, 5.0], 3);
        let original =
            AipwAte::crossfit_problem(&est.prepare(&data, &estimand, &query).unwrap(), &ctx());
        let idx: Vec<usize> = (0..original.nrows).collect();
        let same = resampled_problem(&original, &idx, None);
        let (a, fit_a) = est.crossfit_table(&original, false, &ctx(), None).unwrap();
        let (b, fit_b) = est.crossfit_table(&same, false, &ctx(), None).unwrap();
        let bits = |t: &crate::scores::ScoreTable| -> Vec<u64> {
            t.scores.iter().map(|v| v.to_bits()).collect()
        };
        assert_eq!(bits(&a), bits(&b));
        assert_eq!(fit_a.selections, fit_b.selections);
    }

    /// Per-replicate draws on the penalized stream match the replicate-major batch.
    #[test]
    fn penalized_bootstrap_indexes_match_the_batch_stream() {
        let n = 9usize;
        let replicates = 6usize;
        let ctx = ExecutionContext::for_tests(23);
        let mut batch = vec![antecedent_data::GlobalId::new(0); n * replicates];
        antecedent_data::fill_resample_index_batch(
            antecedent_data::ResamplingPlan::IidBootstrap,
            n,
            replicates,
            None,
            &ctx,
            super::PENALIZED_BOOTSTRAP_STREAM,
            &mut batch,
        )
        .unwrap();
        let mut ids = Vec::new();
        let mut idx = Vec::new();
        let (mut held_ids, mut held_idx) = (0usize, 0usize);
        for replicate in 0..replicates {
            super::fill_penalized_bootstrap_indexes(
                n,
                u64::try_from(replicate).unwrap(),
                &ctx,
                &mut ids,
                &mut idx,
                None,
                None,
                crate::util::SamplingDeclaration::Undeclared,
            )
            .unwrap();
            let expect: Vec<usize> = batch[replicate * n..(replicate + 1) * n]
                .iter()
                .map(|id| usize::try_from(id.get()).unwrap())
                .collect();
            assert_eq!(idx, expect);
            if replicate == 0 {
                held_ids = ids.capacity();
                held_idx = idx.capacity();
            } else {
                assert_eq!(ids.capacity(), held_ids);
                assert_eq!(idx.capacity(), held_idx);
            }
        }
    }

    /// The refit bootstrap clones only to install a different fold seed or propensity.
    #[test]
    fn penalized_bootstrap_base_clones_only_when_the_seed_or_propensity_differs() {
        let (data, estimand) = confounded_scm(40, 3);
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let est = ridge_estimator(&[0.5, 5.0], 2);
        let problem = est.prepare(&data, &estimand, &query).unwrap();
        let nuisance = est.propensity.clone();
        assert_eq!(problem.fold_seed, 0);
        assert_eq!(problem.propensity, nuisance);

        let changed = super::penalized_bootstrap_base(&problem, 7, nuisance.clone());
        assert!(matches!(changed, std::borrow::Cow::Owned(_)));
        assert_eq!(changed.fold_seed, 7);
        assert_eq!(changed.propensity, nuisance);

        let mut seeded = problem.clone();
        seeded.fold_seed = 7;
        let same = super::penalized_bootstrap_base(&seeded, 7, nuisance);
        assert!(matches!(same, std::borrow::Cow::Borrowed(_)));
        assert!(std::ptr::eq(same.as_ref(), &seeded));

        let swapped = super::penalized_bootstrap_base(&seeded, 7, PropensityNuisance::default());
        assert!(matches!(swapped, std::borrow::Cow::Owned(_)));
        assert_eq!(swapped.fold_seed, 7);
        assert_eq!(swapped.propensity, PropensityNuisance::default());
    }

    /// The refit bootstrap's replicate selects every fold's penalty on the resample only.
    ///
    /// The resample draws only from the first half of the rows, repeating them. (1) The
    /// penalty of each fold equals one re-derived here from that fold's training rows of the
    /// resample alone (the resample's own fold plan, designs, treatment and unit identities).
    /// (2) Rewriting every row the resample did not draw (covariates, treatment, outcome)
    /// leaves every selection, and therefore the replicate, unchanged.
    #[test]
    fn a_bootstrap_replicate_selects_its_penalty_on_the_resample_only() {
        use crate::propensity::RidgeFoldInput;
        let (data, estimand) = confounded_scm(300, 73);
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let tuning = crate::propensity::RidgeTuning::new(&[0.1, 1.0, 10.0, 100.0], 3).unwrap();
        let est = ridge_estimator(&[0.1, 1.0, 10.0, 100.0], 3);
        let original =
            AipwAte::crossfit_problem(&est.prepare(&data, &estimand, &query).unwrap(), &ctx());
        let n = original.nrows;
        let idx: Vec<usize> = (0..n).map(|k| (k * 7) % (n / 2)).collect();
        let replicate = resampled_problem(&original, &idx, None);
        let (_, fit) = est.crossfit_table(&replicate, false, &ctx(), None).unwrap();
        assert_eq!(fit.selections.len(), 5);

        let arms: Vec<u32> = replicate.treatment.iter().map(|&t| u32::from(t > 0.5)).collect();
        let folds = crate::crossfit_plan::crossfit_fold_plan(
            &arms,
            &replicate.row_index,
            5,
            replicate.fold_seed,
        )
        .unwrap();
        for selection in &fit.selections {
            let (train, valid): (Vec<usize>, Vec<usize>) =
                (0..n).partition(|&i| folds[i] as usize != selection.fold);
            let (mut design_train, mut design_valid) = (Vec::new(), Vec::new());
            let ncols = replicate.design_ncols;
            select_rows_colmajor(&replicate.design_matrix, n, ncols, &train, &mut design_train);
            select_rows_colmajor(&replicate.design_matrix, n, ncols, &valid, &mut design_valid);
            let t_train = gather(&replicate.treatment, &train);
            let units: Vec<_> = train.iter().map(|&i| replicate.row_index[i]).collect();
            let (_, again) = crate::propensity::fit_ridge_fold(
                &tuning,
                &RidgeFoldInput {
                    design_train: &design_train,
                    n_train: train.len(),
                    design_valid: &design_valid,
                    n_valid: valid.len(),
                    ncols,
                    t_train: &t_train,
                    units: &units,
                    fold: selection.fold,
                    seed: replicate.fold_seed,
                    weights: None,
                },
                &ctx(),
            )
            .unwrap();
            assert_eq!(
                again.lambda.to_bits(),
                selection.lambda.to_bits(),
                "fold {}",
                selection.fold
            );
        }

        let ncols = original.design_ncols;
        let mut design = original.design_matrix.to_vec();
        for c in 1..ncols {
            for i in n / 2..n {
                design[c * n + i] += 5.0;
            }
        }
        let mut rewritten = original.clone();
        rewritten.design_matrix = design.into();
        rewritten.treatment = original
            .treatment
            .iter()
            .enumerate()
            .map(|(i, &t)| if i >= n / 2 { 1.0 - t } else { t })
            .collect::<Vec<_>>()
            .into();
        rewritten.outcome = original
            .outcome
            .iter()
            .enumerate()
            .map(|(i, &y)| if i >= n / 2 { y + 40.0 } else { y })
            .collect::<Vec<_>>()
            .into();
        let replicate_b = resampled_problem(&rewritten, &idx, None);
        let (_, fit_b) = est.crossfit_table(&replicate_b, false, &ctx(), None).unwrap();
        assert_eq!(fit_b.selections, fit.selections);
    }

    #[test]
    fn aipw_bootstrap_follows_the_declared_sampling_design() {
        use antecedent_core::DeclaredSamplingDesign::IndependentSubjectTrajectories;
        let (data, estimand) = confounded_scm(400, 12);
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let base = AipwAte { bootstrap_replicates: 6, ..AipwAte::new() };

        // Undeclared, the rows are one-row subjects and the row draw is the default.
        let prep = base.prepare(&data, &estimand, &query).unwrap();
        base.fit(&prep, &mut AipwWorkspace::default(), &ctx(), AssumptionSet::new()).unwrap();

        // A declared trajectory design with no subject labels is a row-level draw.
        let declared = base.clone().with_sampling_design(IndependentSubjectTrajectories);
        let prep = declared.prepare(&data, &estimand, &query).unwrap();
        let refused =
            declared.fit(&prep, &mut AipwWorkspace::default(), &ctx(), AssumptionSet::new());
        assert!(
            matches!(
                refused,
                Err(EstimationError::Data(antecedent_data::DataError::SamplingDesign(_)))
            ),
            "{refused:?}"
        );

        // With subject labels the whole trajectory is the draw, and the design allows it.
        let subjects: Vec<u32> = (0..400).map(|i| u32::try_from(i / 4).unwrap_or(0)).collect();
        let trajectories =
            AipwAte { se_kind: AnalyticSeKind::Cluster, cluster_ids: Some(subjects), ..declared };
        let prep = trajectories.prepare(&data, &estimand, &query).unwrap();
        let fit = trajectories
            .fit(&prep, &mut AipwWorkspace::default(), &ctx(), AssumptionSet::new())
            .unwrap();
        assert!(fit.se_bootstrap().is_some());
    }
}
