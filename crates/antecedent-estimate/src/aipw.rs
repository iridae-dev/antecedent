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
use crate::arm::TreatmentArm;
use crate::bootstrap_wave::retained_index_slots;
use crate::cluster_dml_aipw::ClusterDml;
use crate::error::EstimationError;
use crate::inference::InferenceSpec;
use crate::overlap::{IpwTarget, OverlapPolicy, OverlapReport};
use crate::propensity::{
    OtherLevels, PenalizedReport, PenalizedVarianceReport, PreparedPropensityProblem,
    PropensityModel, PropensityNuisance, REFIT_BOOTSTRAP_UNCERTAINTY_KIND, ReplicatePenalties,
    clamp_scores, clip_of, crossfit_influence_se, default_propensity_overlap, fold_supports,
    gather, gather_into, prepare_propensity_problem_with_levels, split_by_treatment, trim_of,
    trim_retained_rows,
};
use crate::sampling_weights::{RetainedSamplingWeights, SamplingWeights};
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

mod bootstrap;
mod checked;
mod nuisance;
mod scores;

pub(crate) use bootstrap::{resampled_problem, select_rows_colmajor};
pub use checked::{CheckedAipwLowering, CheckedAipwPreparation, CheckedAipwProcedure};
pub(crate) use nuisance::{fit_outcome_models, fit_outcome_models_weighted, predict_colmajor};

use bootstrap::*;
use nuisance::*;
use scores::*;

/// Doubly robust (AIPW) ATE / ATT / ATC estimator.
///
/// Supports [`TargetPopulation::AllObserved`], [`TargetPopulation::Treated`] (ATT), and
/// [`TargetPopulation::Untreated`] (ATC). Positivity is mandatory:
/// [`OverlapPolicy::ExplicitOverride`] is refused.
#[derive(Clone, Debug)]
pub struct AipwAte {
    /// Dense linear-algebra backend used for the propensity IRLS fit and outcome OLS fits.
    pub backend: FaerBackend,
    /// Inference declarations: bootstrap size and interval request, analytic standard-error
    /// kind, cluster / multiway / panel labels and the sampling design of the rows.
    ///
    /// A cluster-DML declaration ([`Self::cluster_dml`]) requires
    /// `inference.bootstrap_replicates == 0`. With it, `inference.source_cluster_ids` are the
    /// first-endpoint unit labels, mapped to the complete-case rows before the cross-fit plan
    /// is built. The wave-procedure bootstrap interval (`inference.bootstrap_interval`) runs
    /// for the non-penalized bootstrap; the penalized refit bootstrap, which records
    /// per-replicate penalties, keeps its fixed count.
    pub inference: InferenceSpec,
    /// Overlap policy; must be [`OverlapPolicy::RequireDiagnostics`].
    pub overlap: OverlapPolicy,
    /// GLM fitting options for the propensity model.
    pub glm_options: GlmOptions,
    /// Second-endpoint labels of a dyadic [`ClusterDml`] declaration for every row of the
    /// source table. Exclusive with [`Self::cluster_ids_second`].
    pub source_cluster_ids_second: Option<Vec<u32>>,
    /// Second-endpoint labels of a dyadic [`ClusterDml`] declaration (aligned to prepared
    /// rows); `inference.cluster_ids` are then the first endpoints.
    pub cluster_ids_second: Option<Vec<u32>>,
    /// Optional bindings for named predicates / custom target distributions.
    pub population_registry: Option<PopulationRegistry>,
    /// Declared binary-propensity nuisance: the unpenalized logistic by default, or an
    /// explicit ridge-logistic penalty (distinct from `glm_options.ridge_on_separation`).
    pub propensity: PropensityNuisance,
    /// Declared cluster-DML independence unit over `inference.cluster_ids`: whole clusters are
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
}

/// Physical table indices of row identities, refusing an identity the address space cannot hold.
fn physical_indices(ids: &[antecedent_data::GlobalId]) -> Result<Vec<usize>, EstimationError> {
    ids.iter()
        .map(|id| {
            usize::try_from(id.get()).map_err(|_| {
                EstimationError::data_msg("a row identity exceeds the addressable table size")
            })
        })
        .collect()
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
            inference: InferenceSpec::new(200),
            overlap: default_propensity_overlap(),
            glm_options: GlmOptions::default(),
            source_cluster_ids_second: None,
            cluster_ids_second: None,
            population_registry: None,
            propensity: PropensityNuisance::default(),
            cluster_dml: None,
            other_levels: OtherLevels::default(),
            sampling_weights: None,
        }
    }

    crate::inference::with_bootstrap_interval!(inference);

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

    /// Declare the independence unit of the cross-fit: whole clusters of [`InferenceSpec::cluster_ids`]
    /// (which this does not set) share a fold, and the route publishes the cross-fitted point
    /// estimate and score table with no interval. Licensed for the untrimmed `AllObserved`
    /// mean ATE with `bootstrap_replicates == 0`; see [`ClusterDml`].
    #[must_use]
    pub const fn with_cluster_dml(mut self, cluster_dml: ClusterDml) -> Self {
        self.cluster_dml = Some(cluster_dml);
        self
    }

    crate::inference::with_bootstrap_replicates!(inference);

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

    crate::inference::with_se_kind!(inference);

    crate::inference::with_cluster_ids!(inference);

    crate::inference::with_source_cluster_ids!(inference);

    crate::inference::with_sampling_design!(inference);

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
            (self.inference.cluster_ids.as_deref(), self.inference.source_cluster_ids.as_deref()),
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

    crate::inference::with_multiway_ids!(inference);

    crate::inference::with_panel_times!(inference);

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
        if let Some(source) = self.inference.source_cluster_ids.as_deref() {
            crate::se::require_source_labels(
                source,
                self.inference.cluster_ids.as_deref(),
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
                self.inference.bootstrap_replicates,
                self.inference.se_kind,
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
                self.inference.bootstrap_replicates,
                self.inference.se_kind,
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
        )?;
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
            se_kind: self.inference.se_kind,
            bootstrap_replicates: self.inference.bootstrap_replicates,
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
            || checked.lowering.se_kind != self.inference.se_kind
            || checked.lowering.bootstrap_replicates != self.inference.bootstrap_replicates
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
        self.validate_checked_configuration(checked)?;
        self.fit(&checked.problem, workspace, ctx, checked.required_assumptions.clone())
    }

    /// Execute only the point and analytic uncertainty of the checked procedure.
    ///
    /// The receipt keeps its complete inference declaration. Call [`Self::attach_bootstrap`]
    /// with this same fitter, receipt's problem and workspace at the uncertainty boundary;
    /// the point fit is not repeated.
    ///
    /// # Errors
    /// A changed checked configuration, unsupported target, or numerical fit failure.
    pub fn fit_checked_point(
        &self,
        checked: &CheckedAipwPreparation,
        workspace: &mut AipwWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<EffectEstimate, EstimationError> {
        self.validate_checked_configuration(checked)?;
        self.fit_point(&checked.problem, workspace, ctx, checked.required_assumptions.clone())
            .map(|point| point.with_treatment_levels(checked.problem.treatment_levels))
    }

    fn validate_checked_configuration(
        &self,
        checked: &CheckedAipwPreparation,
    ) -> Result<(), EstimationError> {
        if checked.lowering.se_kind != self.inference.se_kind
            || checked.lowering.bootstrap_replicates != self.inference.bootstrap_replicates
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
        Ok(())
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
            || self.inference.cluster_ids.is_some()
            || self.inference.source_cluster_ids.is_some()
            || self.inference.multiway_ids.is_some()
            || !matches!(
                self.inference.se_kind,
                AnalyticSeKind::Homoskedastic | AnalyticSeKind::Hc1
            )
            || self.inference.bootstrap_replicates != 0
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
            || self.inference.cluster_ids.is_some()
            || self.inference.source_cluster_ids.is_some()
            || self.inference.multiway_ids.is_some()
            || !matches!(
                self.inference.se_kind,
                AnalyticSeKind::Homoskedastic | AnalyticSeKind::Hc1
            )
            || self.inference.bootstrap_replicates != 0
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
        if self.inference.bootstrap_replicates == 0 {
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

    /// One point fit: check the route, then dispatch to the custom-population, cross-fitted or
    /// full-sample stages.
    fn fit_point(
        &self,
        problem: &PreparedPropensityProblem,
        workspace: &mut AipwWorkspace,
        ctx: &ExecutionContext,
        assumptions: AssumptionSet,
    ) -> Result<EffectEstimate, EstimationError> {
        self.require_point_route(problem)?;
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
        self.fit_full_sample(problem, workspace, assumptions)
    }

    /// Refuse a problem prepared under a different nuisance or cluster-DML declaration than this
    /// estimator's, a penalized or cluster-DML route outside its licensed scope, and a target
    /// population the AIPW score does not cover.
    fn require_point_route(
        &self,
        problem: &PreparedPropensityProblem,
    ) -> Result<(), EstimationError> {
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
                self.inference.bootstrap_replicates,
                self.inference.se_kind,
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
        Ok(())
    }

    /// The full-sample fit, in stages: fit the propensity and select the contrast rows, fit
    /// both outcome arms and score, centre or nuisance-correct the influence values, and
    /// assemble the estimate with its standard error.
    fn fit_full_sample(
        &self,
        problem: &PreparedPropensityProblem,
        workspace: &mut AipwWorkspace,
        assumptions: AssumptionSet,
    ) -> Result<EffectEstimate, EstimationError> {
        let model = PropensityModel::fit(
            problem,
            self.backend,
            &mut workspace.propensity,
            &self.glm_options,
        )
        .map_err(|error| self.propensity.record_failed_fit(error))?;
        let rows = ContrastRows::select(problem, &model)?;
        let ate = self.score_contrast(problem, &rows, workspace)?;
        self.adjust_influence(problem, &model, &rows, workspace, ate)?;
        self.assemble_full_sample(problem, &model, &rows, workspace, ate, assumptions)
    }

    /// Fit both outcome arms on the contrast rows and score the contrast into
    /// `workspace.psi`; the returned value is the AIPW estimate (the mean score).
    fn score_contrast(
        &self,
        problem: &PreparedPropensityProblem,
        rows: &ContrastRows<'_>,
        workspace: &mut AipwWorkspace,
    ) -> Result<f64, EstimationError> {
        let ncols = problem.design_ncols;
        let nrows = rows.treatment.len();
        let used_weights = rows.weights();
        let (beta0, beta1) = fit_outcome_models_weighted(
            &rows.design,
            nrows,
            ncols,
            &rows.treatment,
            &rows.outcome,
            used_weights,
            self.backend,
            workspace,
        )?;
        predict_colmajor(&rows.design, nrows, ncols, &beta0, &mut workspace.mu0);
        predict_colmajor(&rows.design, nrows, ncols, &beta1, &mut workspace.mu1);
        aipw_psi_weighted(
            &rows.treatment,
            &rows.outcome,
            &rows.propensity,
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
        Ok(workspace.psi.iter().sum::<f64>() / workspace.psi.len() as f64)
    }

    /// Turn the scores in `workspace.psi` into the influence function the variance reads:
    /// the ATT / ATC centred score, or the ATE score projected off the nuisance estimators.
    fn adjust_influence(
        &self,
        problem: &PreparedPropensityProblem,
        model: &PropensityModel,
        rows: &ContrastRows<'_>,
        workspace: &mut AipwWorkspace,
        ate: f64,
    ) -> Result<(), EstimationError> {
        let used_weights = rows.weights();
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
                &rows.treatment,
                &problem.target_population,
                ate,
                used_weights,
            );
            return Ok(());
        }
        let e_raw: Cow<'_, [f64]> = match &rows.retained {
            Some(idx) => Cow::Owned(gather(&model.fit.scores, idx)),
            None => Cow::Borrowed(model.fit.scores.as_slice()),
        };
        let fit = AipwNuisanceFit {
            treatment: &rows.treatment,
            outcome: &rows.outcome,
            e_used: &rows.propensity,
            e_raw: &e_raw,
            mu0: &workspace.mu0,
            mu1: &workspace.mu1,
            design: &rows.design,
            ncols: problem.design_ncols,
            weights: used_weights,
        };
        correct_aipw_psi_for_nuisances(&mut workspace.psi, &fit)
    }

    /// The standard error of the influence values and the estimate that carries it, with the
    /// overlap report and the sampling-weight record.
    fn assemble_full_sample(
        &self,
        problem: &PreparedPropensityProblem,
        model: &PropensityModel,
        rows: &ContrastRows<'_>,
        workspace: &AipwWorkspace,
        ate: f64,
        assumptions: AssumptionSet,
    ) -> Result<EffectEstimate, EstimationError> {
        let source_rows = self.source_rows(problem)?;
        let declared_clusters = crate::se::declared_row_labels(
            self.inference.cluster_ids.as_deref(),
            self.inference.source_cluster_ids.as_deref(),
            source_rows.iter().copied(),
        )?;
        let se_analytic = crate::se::influence_se_kind(
            self.inference.se_kind,
            &workspace.psi,
            problem.nrows,
            declared_clusters.as_deref(),
            self.inference.multiway_ids.as_deref(),
            self.inference.panel_times.as_deref(),
            rows.retained.as_deref(),
        )?;
        let overlap_report = Some(crate::propensity::propensity_overlap_report(
            problem,
            &model.fit.scores,
            None,
            IpwTarget::from_population(&problem.target_population).ok(),
        ));
        Ok(EffectEstimate::new(ate, se_analytic, assumptions, problem.overlap)
            .with_preparation(Some(problem.preparation_summary()))
            .with_se_kind(self.inference.se_kind)
            .with_sampling_weights(rows.sampling.as_ref().map(|s| s.record))
            .with_overlap_report(overlap_report)
            .with_influence(Some(Arc::from(workspace.psi.as_slice()))))
    }

    fn fit_custom(
        &self,
        problem: &PreparedPropensityProblem,
        ctx: &ExecutionContext,
        assumptions: AssumptionSet,
    ) -> Result<EffectEstimate, EstimationError> {
        if trim_of(problem.overlap).is_some()
            || !matches!(self.inference.se_kind, AnalyticSeKind::Homoskedastic)
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
    ///
    /// Stages: build the cross-fitted score table, summarize and contrast it, take the standard
    /// error, assemble the estimate, then record the penalized and cluster-DML routes.
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
        let iid_se = matches!(self.inference.se_kind, AnalyticSeKind::Homoskedastic);
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
        let se_analytic =
            if iid_se { contrast.se } else { self.robust_influence_se(problem, &influence)? };
        let overlap_report = treated_arm_overlap_report(&table, problem);
        let cluster_se = self.cluster_dml_se(problem, &table, &influence)?;
        let mut result =
            EffectEstimate::new(contrast.value, se_analytic, assumptions, problem.overlap)
                .with_preparation(Some(problem.preparation_summary()))
                .with_se_kind(self.inference.se_kind)
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
            self.record_cluster_dml(&mut result, n_folds, fold_seed, selections, cluster_se);
        }
        Ok(result)
    }

    /// The dependence-robust analytic standard error of the contrast's influence values under
    /// the declared cluster, multiway or panel structure.
    fn robust_influence_se(
        &self,
        problem: &PreparedPropensityProblem,
        influence: &[f64],
    ) -> Result<f64, EstimationError> {
        let source_rows = self.source_rows(problem)?;
        let declared_clusters = crate::se::declared_row_labels(
            self.inference.cluster_ids.as_deref(),
            self.inference.source_cluster_ids.as_deref(),
            source_rows.iter().copied(),
        )?;
        crate::se::influence_se_kind(
            self.inference.se_kind,
            influence,
            problem.nrows,
            declared_clusters.as_deref(),
            self.inference.multiway_ids.as_deref(),
            self.inference.panel_times.as_deref(),
            None,
        )
    }

    /// Whole-cluster folds: the sandwich standard error of the same scores, under the declared
    /// independence unit (`docs/guides/clustered-dml.md`), with its reference degrees of
    /// freedom. `None` without a cluster-DML declaration.
    fn cluster_dml_se(
        &self,
        problem: &PreparedPropensityProblem,
        table: &crate::scores::ScoreTable,
        influence: &[f64],
    ) -> Result<Option<(f64, usize)>, EstimationError> {
        let (first_units, second_units) = if self.cluster_dml.is_some() {
            self.unit_labels(&problem.row_index)?
        } else {
            (None, None)
        };
        match (self.cluster_dml, first_units.as_deref()) {
            (Some(spec), Some(labels)) => {
                let receipt = spec.receipt_with_influence(
                    table,
                    labels,
                    second_units.as_deref(),
                    problem.sampling_weights.is_some().then_some(influence),
                )?;
                Ok(Some((receipt.cluster_sandwich_se, receipt.reference_df)))
            }
            _ => Ok(None),
        }
    }

    /// Re-label a cross-fitted estimate as a cluster-DML one: the cluster sandwich replaces the
    /// iid standard error, and the iid covariance and influence values stay withheld because
    /// their dependence is wrong for the declared unit.
    fn record_cluster_dml(
        &self,
        result: &mut EffectEstimate,
        n_folds: usize,
        fold_seed: u64,
        selections: Vec<crate::propensity::RidgeFoldSelection>,
        cluster_se: Option<(f64, usize)>,
    ) {
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
        result.crossfit.learner_provenance = selections.into_iter().map(|s| s.provenance).collect();
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
}

#[cfg(test)]
mod tests;
