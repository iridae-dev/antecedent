//! Generalized linear model (logistic) adjustment ATE estimator for binary outcomes.
//!
//! Fits a logistic GLM `Y ~ T + Z` and recovers the ATE by finite-difference g-computation:
//! the fitted model is evaluated at `T = active` and `T = control` for every row (holding `Z`
//! fixed), and the ATE is the mean of the per-row predicted-probability contrast. This is the
//! standard g-computation contrast for a non-identity link, since the coefficient on `T` alone
//! is a log-odds-ratio, not a probability-scale effect.
//!
//! Positivity is handled the same way as [`crate::adjustment::LinearAdjustmentAte`]:
//! [`OverlapPolicy::ExplicitOverride`] is the only supported policy, since this is a regression
//! (not propensity-based) path.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![cfg_attr(
    test,
    allow(
        clippy::cast_possible_truncation,
        clippy::float_cmp,
        reason = "test fixtures compare exact constants and index with small literals"
    )
)]

use std::sync::Arc;

use antecedent_core::{
    AssumptionSet, AverageEffectQuery, ExecutionContext, TargetPopulation, VariableId,
};
use antecedent_data::{TableView, TabularData};
use antecedent_expr::IdentifiedEstimand;
use antecedent_stats::{
    CompiledDesign, FaerBackend, GlmDesignRef, GlmFamily, GlmOptions, LeastSquaresWorkspace,
    fit_glm, fit_glm_weighted, score_coefficient_covariance,
};

use crate::adjustment::{EffectEstimate, intervention_f64};
use crate::arm::TreatmentArm;
use crate::error::EstimationError;
use crate::gcomp::{GcompContrast, linear_predictors};
use crate::inference::InferenceSpec;
use crate::overlap::OverlapPolicy;
use crate::sampling_weights::{RetainedSamplingWeights, SamplingWeights};
use crate::se::{AnalyticSeKind, SeSpec};
use crate::util::BootstrapOutcome;

mod family;
mod variance;

pub(crate) use family::{FamilyKernel, fisher_weight, mean_derivative, resolve_nb_alpha};
pub(crate) use variance::{adjustment_glm_covariance, solve_adjustment_glm};

use variance::*;

/// Prepared GLM adjustment problem (compiled design retained).
#[derive(Clone, Debug)]
pub struct PreparedGlmProblem {
    /// Compiled `[1 | T | Z…]` design; outcome must be binary (0/1).
    pub design: CompiledDesign,
    /// Estimand method tag.
    pub method: Arc<str>,
    /// Adjustment set.
    pub adjustment_set: Arc<[VariableId]>,
    /// Overlap policy applied.
    pub overlap: OverlapPolicy,
    /// Active treatment level used for the g-computation contrast.
    pub active: f64,
    /// Control treatment level used for the g-computation contrast.
    pub control: f64,
    /// GLM family used for this problem.
    pub family: GlmFamily,
    /// Target population for g-computation averaging.
    pub target_population: TargetPopulation,
    /// Complete-case treatment values.
    pub treatment: Arc<[f64]>,
    /// Sampling weights of the retained rows (mean one) and what they did to this problem.
    /// The design stays unscaled; the weighted fit, g-computation average and sandwich read
    /// these per-row weights directly.
    pub sampling: Option<RetainedSamplingWeights>,
    /// Preparation receipt for the finite table this problem was read from.
    pub preparation: antecedent_data::materialize::PreparationReceipt,
}

/// Estimation workspace (reusable across bootstrap replicates).
#[derive(Clone, Debug, Default)]
pub struct GlmAdjustmentWorkspace {
    /// IRLS least-squares scratch.
    pub ols: LeastSquaresWorkspace,
}

/// Logistic GLM adjustment estimator for binary-outcome backdoor ATE.
///
/// ATE is estimated by finite-difference g-computation: `mean(μ(T=active, Z) − μ(T=control, Z))`
/// over the complete-case sample, where `μ` is the fitted logistic mean function.
#[derive(Clone, Debug)]
pub struct GlmAdjustmentAte {
    /// Dense linear-algebra backend used by the IRLS inner loop.
    pub backend: FaerBackend,
    /// Inference declarations: bootstrap size and interval request, analytic standard-error
    /// kind, cluster / multiway / panel labels and the sampling design of the rows.
    pub inference: InferenceSpec,
    /// Overlap policy (must be [`OverlapPolicy::ExplicitOverride`]).
    pub overlap: OverlapPolicy,
    /// GLM fitting options (max iterations, convergence tolerance).
    pub glm_options: GlmOptions,
    /// Outcome family / link.
    pub family: GlmFamily,
    /// Registry for named [`TargetPopulation::Predicate`] selections.
    pub population_registry: Option<antecedent_core::PopulationRegistry>,
    /// Sampling weights of every row of the source table (see [`crate::sampling_weights`]).
    /// The family is fit by prior-weighted IRLS, the g-computation contrast is averaged under
    /// the weights, and the standard error is the weighted score sandwich (the classical
    /// Fisher delta method treats weights as precision and is refused). Logit, probit,
    /// Poisson, negative-binomial (weighted dispersion update) and Gaussian families.
    pub sampling_weights: Option<SamplingWeights>,
}

impl Default for GlmAdjustmentAte {
    fn default() -> Self {
        Self::new()
    }
}

impl GlmAdjustmentAte {
    /// Default: 200 bootstrap replicates, explicit overlap override, ridge-on-separation.
    #[must_use]
    pub fn new() -> Self {
        Self {
            backend: FaerBackend,
            inference: InferenceSpec::new(200),
            overlap: OverlapPolicy::ExplicitOverride,
            glm_options: GlmOptions::default(),
            family: GlmFamily::BinomialLogit,
            population_registry: None,
            sampling_weights: None,
        }
    }

    crate::inference::with_bootstrap_interval!(inference);

    /// Declare sampling weights for every row of the source table.
    #[must_use]
    pub fn with_sampling_weights(mut self, weights: SamplingWeights) -> Self {
        self.sampling_weights = Some(weights);
        self
    }

    crate::inference::with_bootstrap_replicates!(inference);

    /// Set the overlap policy. Must remain [`OverlapPolicy::ExplicitOverride`] — this is a
    /// regression (not propensity-based) path.
    #[must_use]
    pub const fn with_overlap(mut self, overlap: OverlapPolicy) -> Self {
        self.overlap = overlap;
        self
    }

    /// Set the GLM fitting options (max iterations, convergence tolerance).
    #[must_use]
    pub const fn with_glm_options(mut self, glm_options: GlmOptions) -> Self {
        self.glm_options = glm_options;
        self
    }

    /// Set the outcome family / link (default [`GlmFamily::BinomialLogit`]).
    #[must_use]
    pub const fn with_family(mut self, family: GlmFamily) -> Self {
        self.family = family;
        self
    }

    crate::inference::with_se_kind!(inference);

    crate::inference::with_cluster_ids!(inference);

    crate::inference::with_source_cluster_ids!(inference);

    crate::inference::with_sampling_design!(inference);

    crate::inference::with_multiway_ids!(inference);

    crate::inference::with_panel_times!(inference);

    /// Prepare design from tabular data, identified estimand, and query levels.
    ///
    /// Accepts `backdoor.adjustment` / `backdoor.efficient` estimands.
    ///
    /// # Errors
    ///
    /// Overlap policy is not `ExplicitOverride`, incompatible estimand, unsupported query, or
    /// missing/invalid data columns.
    #[allow(
        clippy::float_cmp,
        reason = "the two arms are user-supplied intervention levels, and identical levels are the degenerate contrast this exact-equality check rejects; the binomial outcome is coded exactly 0.0 or 1.0, so exact comparison is the intended test"
    )]
    #[expect(
        clippy::too_many_lines,
        reason = "one linear preparation: validate the request, resolve columns, then assemble the design"
    )]
    pub fn prepare(
        &self,
        data: &TabularData,
        estimand: &IdentifiedEstimand,
        query: &AverageEffectQuery,
    ) -> Result<PreparedGlmProblem, EstimationError> {
        crate::util::require_explicit_override(
            self.overlap,
            "GlmAdjustmentAte requires ExplicitOverride overlap policy",
        )?;
        if !estimand.is_adjustment_shaped() {
            return Err(EstimationError::IncompatibleEstimand {
                message: "GlmAdjustmentAte expects an adjustment-shaped estimand",
            });
        }
        query.validate()?;
        if !query.effect_modifiers.is_empty() {
            return Err(EstimationError::unsupported_coded(
                antecedent_core::reason_code!("option_not_applicable"),
                "GLM adjustment does not support effect modifiers",
            ));
        }
        if !matches!(
            query.target_population,
            TargetPopulation::AllObserved
                | TargetPopulation::Treated
                | TargetPopulation::Untreated
                | TargetPopulation::Predicate(_)
        ) {
            return Err(EstimationError::unsupported_coded(
                antecedent_core::reason_code!("route_not_supported"),
                "GLM adjustment supports AllObserved, Treated, Untreated, or Predicate",
            ));
        }
        if let Some(source) = self.inference.source_cluster_ids.as_deref() {
            crate::se::require_source_labels(
                source,
                self.inference.cluster_ids.as_deref(),
                data.row_count(),
            )?;
        }
        let treatment = query.treatment;
        let outcome = query.outcome;
        let active = intervention_f64(&query.active)?;
        let control = intervention_f64(&query.control)?;
        if active == control {
            return Err(EstimationError::unsupported(
                "active and control treatment levels must differ",
            ));
        }

        let mut ids = Vec::with_capacity(2 + estimand.adjustment_set.len());
        ids.push(treatment);
        ids.push(outcome);
        ids.extend_from_slice(&estimand.adjustment_set);
        let crate::prepare::AdmittedColumns { data, mut row_mask, receipt: preparation } =
            crate::prepare::admit_estimator_columns(data, &ids, crate::prepare::FINITE_TABLE_ONLY)?;
        crate::prepare::intersect_predicate_mask(
            &mut row_mask,
            &query.target_population,
            data.row_count(),
            self.population_registry.as_ref(),
        )?;
        if self.sampling_weights.is_some()
            && matches!(self.inference.se_kind, AnalyticSeKind::Homoskedastic)
        {
            return Err(crate::sampling_weights::unsupported_route(
                "the Fisher delta-method standard error treats weights as precision and is \
                 invalid for sampling weights; request a heteroskedasticity-robust kind \
                 (hc0..hc3) or the cluster sandwich",
            ));
        }
        let sampling = self
            .sampling_weights
            .as_ref()
            .map(|weights| weights.restrict(&mut row_mask))
            .transpose()?;
        let design = crate::prepare::direct_adjustment_design(
            data,
            treatment,
            outcome,
            &estimand.adjustment_set,
            &[],
            &row_mask,
        )?;
        let t: Arc<[f64]> = Arc::from(&design.matrix[design.nrows..2 * design.nrows]);
        let y = &design.outcome;
        match self.family {
            GlmFamily::BinomialLogit | GlmFamily::BinomialProbit => {
                for &yi in y.iter() {
                    if !(yi == 0.0 || yi == 1.0) {
                        return Err(EstimationError::unsupported(
                            "Binomial GlmAdjustmentAte requires a binary (0/1) outcome",
                        ));
                    }
                }
            }
            GlmFamily::PoissonLog | GlmFamily::NegativeBinomial => {
                for &yi in y.iter() {
                    if !(yi.is_finite() && yi >= 0.0) {
                        return Err(EstimationError::unsupported(
                            "Poisson/NB GlmAdjustmentAte requires non-negative outcomes",
                        ));
                    }
                }
            }
            GlmFamily::GaussianIdentity => {}
        }
        Ok(PreparedGlmProblem {
            design,
            method: Arc::clone(&estimand.method),
            adjustment_set: Arc::clone(&estimand.adjustment_set),
            overlap: self.overlap,
            active,
            control,
            family: self.family,
            target_population: query.target_population.clone(),
            treatment: t,
            sampling,
            preparation,
        })
    }

    /// Fit the logistic GLM and compute the g-computation ATE, with optional bootstrap.
    ///
    /// # Errors
    ///
    /// GLM/backend failure.
    pub fn fit(
        &self,
        problem: &PreparedGlmProblem,
        workspace: &mut GlmAdjustmentWorkspace,
        ctx: &ExecutionContext,
        assumptions: AssumptionSet,
    ) -> Result<EffectEstimate, EstimationError> {
        let point = self.fit_point(problem, workspace, assumptions)?;
        self.attach_bootstrap(problem, workspace, ctx, point)
    }

    /// Attach the bootstrap SE onto a point estimate from [`Self::fit`] (progressive
    /// uncertainty stage). Equal to what `fit` publishes when `bootstrap_replicates > 0`,
    /// without refitting the point model.
    ///
    /// # Errors
    ///
    /// Bootstrap failure.
    pub fn attach_bootstrap(
        &self,
        problem: &PreparedGlmProblem,
        workspace: &mut GlmAdjustmentWorkspace,
        ctx: &ExecutionContext,
        point: EffectEstimate,
    ) -> Result<EffectEstimate, EstimationError> {
        if self.inference.bootstrap_replicates == 0 {
            return Ok(point);
        }
        let t_col = problem
            .design
            .treatment_column()
            .ok_or_else(|| EstimationError::stats_msg("missing treatment column"))?;
        let boot = self.bootstrap_se(problem, workspace, ctx, t_col)?;
        Ok(point.with_bootstrap_outcome(boot))
    }

    fn fit_point(
        &self,
        problem: &PreparedGlmProblem,
        workspace: &mut GlmAdjustmentWorkspace,
        assumptions: AssumptionSet,
    ) -> Result<EffectEstimate, EstimationError> {
        let t_col = problem
            .design
            .treatment_column()
            .ok_or_else(|| EstimationError::stats_msg("missing treatment column"))?;
        let row_weights = problem.sampling.as_ref().map(|s| &*s.weights);
        let design_ref = GlmDesignRef {
            x_colmajor: &problem.design.matrix,
            nrows: problem.design.nrows,
            ncols: problem.design.ncols,
            y: &problem.design.outcome,
        };
        let glm_fit = match row_weights {
            Some(weights) => {
                let fit = fit_glm_weighted(
                    problem.family,
                    design_ref,
                    weights,
                    &self.backend,
                    &mut workspace.ols,
                    &self.glm_options.without_separation_ridge(),
                )
                .map_err(crate::error::EstimationError::from)?;
                fit.require_ok().map_err(crate::error::EstimationError::from)?;
                crate::adjustment_resume::record_model_fit();
                fit
            }
            None => solve_adjustment_glm(
                problem.family,
                design_ref,
                self.backend,
                &mut workspace.ols,
                &self.glm_options,
            )?,
        };

        let contrast = GcompContrast {
            family: problem.family,
            x_colmajor: &problem.design.matrix,
            nrows: problem.design.nrows,
            ncols: problem.design.ncols,
            t_col,
            active: problem.active,
            control: problem.control,
        };
        let diffs = contrast.diffs(&glm_fit.coefficients);
        let target = GcompTarget {
            treatment: &problem.treatment,
            population: &problem.target_population,
            weights: row_weights,
        };
        let ate = average_gcomp_for_target(
            &diffs,
            &problem.treatment,
            &problem.target_population,
            row_weights,
        )?;
        let nb_alpha = resolve_nb_alpha(
            problem.family,
            glm_fit.nb_alpha,
            &problem.design.matrix,
            problem.design.nrows,
            problem.design.ncols,
            &glm_fit.coefficients,
            &problem.design.outcome,
        );
        let retained_clusters = self
            .inference
            .source_cluster_ids
            .as_deref()
            .map(|source| crate::se::retained_row_labels(source, &problem.design.row_selection))
            .transpose()?;
        let cluster_ids = retained_clusters.as_deref().or(self.inference.cluster_ids.as_deref());
        let se_analytic = match self.inference.se_kind {
            AnalyticSeKind::Homoskedastic => gcomp_delta_method_se(
                &contrast,
                &glm_fit.coefficients,
                glm_fit.deviance,
                nb_alpha,
                &target,
            ),
            other => gcomp_sandwich_se(
                SeSpec {
                    kind: other,
                    cluster_ids,
                    multiway_ids: self.inference.multiway_ids.as_deref(),
                    panel_times: self.inference.panel_times.as_deref(),
                },
                &contrast,
                &glm_fit.coefficients,
                &problem.design.outcome,
                nb_alpha,
                &target,
            )?,
        };

        Ok(EffectEstimate::new(ate, se_analytic, assumptions, problem.overlap)
            .with_preparation(Some(crate::prepare::preparation_summary(&problem.preparation)))
            .with_se_kind(self.inference.se_kind)
            .with_sampling_weights(problem.sampling.as_ref().map(|s| s.record)))
    }

    fn bootstrap_se(
        &self,
        problem: &PreparedGlmProblem,
        _workspace: &mut GlmAdjustmentWorkspace,
        ctx: &ExecutionContext,
        t_col: usize,
    ) -> Result<BootstrapOutcome, EstimationError> {
        let n = problem.design.nrows;
        let p = problem.design.ncols;
        let clusters = crate::se::declared_row_labels(
            self.inference.cluster_ids.as_deref(),
            self.inference.source_cluster_ids.as_deref(),
            problem.design.row_selection.iter().copied(),
        )?;
        if let Some(labels) = &clusters {
            crate::se::require_clusters(Some(labels.as_ref()), n)?;
        }
        let series_block = self.inference.panel_times.as_ref().map(|_| {
            let span = match self.inference.se_kind {
                AnalyticSeKind::NeweyWest { lag } | AnalyticSeKind::PanelClusterHac { lag } => {
                    lag.max(1)
                }
                _ => 1,
            };
            antecedent_data::circular_block_length(span, n)
        });
        // Each worker gathers `n * p` design cells and three length-`n` vectors.
        let _gather =
            crate::util::admit_gather_scratch(ctx, n, p + 3, self.inference.bootstrap_replicates)?;
        // Replicates own a cold workspace. A warm IRLS start would change the replicate fit.
        crate::util::bootstrap_outcome_by_design(
            self.inference.bootstrap_interval.as_ref(),
            crate::util::BootstrapDesign {
                clusters: clusters.as_deref(),
                series_block,
                declared: self.inference.sampling_design,
            },
            self.inference.bootstrap_replicates,
            ctx,
            0xC17A_u64,
            n,
            || {
                (
                    GlmAdjustmentWorkspace::default(),
                    vec![0.0; n * p],
                    vec![0.0; n],
                    vec![0.0; n],
                    vec![0.0; n],
                )
            },
            |(ws, x_boot, y_boot, t_boot, w_boot), idx| {
                let m = idx.len();
                x_boot.resize(m * p, 0.0);
                y_boot.resize(m, 0.0);
                t_boot.resize(m, 0.0);
                w_boot.resize(m, 0.0);
                crate::util::gather_bootstrap_vector(y_boot, &problem.design.outcome, idx);
                crate::util::gather_bootstrap_vector(t_boot, &problem.treatment, idx);
                crate::util::gather_cols(x_boot, &problem.design.matrix, n, p, idx);
                let boot_design =
                    GlmDesignRef { x_colmajor: x_boot, nrows: m, ncols: p, y: y_boot };
                let options = self.glm_options.without_separation_ridge();
                let fitted = match problem.sampling.as_ref() {
                    Some(sampling) => {
                        crate::util::gather_bootstrap_vector(w_boot, &sampling.weights, idx);
                        fit_glm_weighted(
                            problem.family,
                            boot_design,
                            w_boot,
                            &self.backend,
                            &mut ws.ols,
                            &options,
                        )
                    }
                    None => {
                        fit_glm(problem.family, boot_design, &self.backend, &mut ws.ols, &options)
                    }
                };
                let Ok(fit) = fitted else {
                    return Ok(None);
                };
                if fit.require_ok().is_err() {
                    return Ok(None);
                };
                let diffs = GcompContrast {
                    family: problem.family,
                    x_colmajor: x_boot,
                    nrows: m,
                    ncols: p,
                    t_col,
                    active: problem.active,
                    control: problem.control,
                }
                .diffs(&fit.coefficients);
                let boot_weights = problem.sampling.as_ref().map(|_| &w_boot[..]);
                match average_gcomp_for_target(
                    &diffs,
                    t_boot,
                    &problem.target_population,
                    boot_weights,
                ) {
                    Ok(ate) => Ok(Some(ate)),
                    Err(_) => Ok(None),
                }
            },
        )
    }
}

#[cfg(test)]
mod tests;
