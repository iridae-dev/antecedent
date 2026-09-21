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

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::similar_names,
    clippy::needless_range_loop
)]

use std::borrow::Cow;
use std::sync::Arc;

use antecedent_core::{
    AssumptionSet, AverageEffectQuery, ExecutionContext, PopulationRegistry, TargetPopulation,
};
use antecedent_data::TabularData;
use antecedent_expr::IdentifiedEstimand;
use antecedent_stats::{
    DenseLinearAlgebra, FaerBackend, GlmOptions, LeastSquaresWorkspace, PropensityWorkspace,
};

use crate::adjustment::EffectEstimate;
use crate::error::EstimationError;
use crate::overlap::{IpwTarget, OverlapPolicy};
use crate::propensity::{
    PreparedPropensityProblem, PropensityModel, clamp_scores, clip_of, default_propensity_overlap,
    gather, gather_into, prepare_propensity_problem_with_registry, split_by_treatment, trim_of,
    trim_retained_rows,
};
use crate::se::AnalyticSeKind;
use crate::util::{BootstrapSeResult, bootstrap_se, stats_err};

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
    /// Optional bindings for named predicates / custom target distributions.
    pub population_registry: Option<PopulationRegistry>,
    /// Multiway cluster ids (one `Vec<u32>` per clustering dimension).
    pub multiway_ids: Option<Vec<Vec<u32>>>,
    /// Optional panel time labels for [`AnalyticSeKind::PanelClusterHac`].
    pub panel_times: Option<Vec<i64>>,
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
            population_registry: None,
            multiway_ids: None,
            panel_times: None,
        }
    }

    /// Set the dense linear-algebra backend used for the propensity IRLS fit and outcome
    /// OLS fits.
    #[must_use]
    pub const fn with_backend(mut self, backend: FaerBackend) -> Self {
        self.backend = backend;
        self
    }

    /// Set the number of bootstrap replicates used for the bootstrap standard error.
    ///
    /// Defaults to 200. Set to `0` to skip bootstrapping and report only the analytic SE.
    /// Each replicate refits both the propensity model and the two outcome models.
    #[must_use]
    pub const fn with_bootstrap_replicates(mut self, replicates: u32) -> Self {
        self.bootstrap_replicates = replicates;
        self
    }

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

    /// Set the analytic standard-error kind (default [`AnalyticSeKind::Homoskedastic`]).
    #[must_use]
    pub const fn with_se_kind(mut self, se_kind: AnalyticSeKind) -> Self {
        self.se_kind = se_kind;
        self
    }

    /// Set cluster ids (aligned to prepared rows) for [`AnalyticSeKind::Cluster`].
    #[must_use]
    pub fn with_cluster_ids(mut self, cluster_ids: Vec<u32>) -> Self {
        self.cluster_ids = Some(cluster_ids);
        self
    }

    /// Set bindings for named predicates / custom target distributions used when the query
    /// targets [`TargetPopulation::Predicate`] or a custom distribution.
    #[must_use]
    pub fn with_population_registry(mut self, registry: PopulationRegistry) -> Self {
        self.population_registry = Some(registry);
        self
    }

    /// Set multiway cluster ids (one `Vec<u32>` per clustering dimension) for
    /// [`AnalyticSeKind::Multiway`].
    #[must_use]
    pub fn with_multiway_ids(mut self, multiway_ids: Vec<Vec<u32>>) -> Self {
        self.multiway_ids = Some(multiway_ids);
        self
    }

    /// Set panel time labels for [`AnalyticSeKind::PanelClusterHac`].
    #[must_use]
    pub fn with_panel_times(mut self, panel_times: Vec<i64>) -> Self {
        self.panel_times = Some(panel_times);
        self
    }

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
        prepare_propensity_problem_with_registry(
            data,
            estimand,
            query,
            self.overlap,
            self.population_registry.as_ref(),
        )
    }

    /// Fit propensity + outcome nuisance models and compute the AIPW effect, with optional
    /// bootstrap.
    ///
    /// # Errors
    ///
    /// Target population other than ATE/ATT/ATC, empty treated/control arm, or GLM/OLS
    /// backend failure.
    #[allow(clippy::too_many_lines)]
    pub fn fit(
        &self,
        problem: &PreparedPropensityProblem,
        workspace: &mut AipwWorkspace,
        ctx: &ExecutionContext,
        assumptions: AssumptionSet,
    ) -> Result<EffectEstimate, EstimationError> {
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
            return self.fit_crossfit_scores(problem, ctx, assumptions, None, false);
        }

        let model = PropensityModel::fit(
            problem,
            &self.backend,
            &mut workspace.propensity,
            &self.glm_options,
        )?;
        // Trim on raw scores: excluded units match the overlap report's common-support claim.
        let retained = trim_retained_rows(&model.fit.scores, trim_of(problem.overlap))?;
        let ncols = problem.design_ncols;
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
        let (beta0, beta1) = fit_outcome_models(
            &design_used,
            nrows,
            ncols,
            &t_used,
            &y_used,
            self.backend,
            workspace,
        )?;
        predict_colmajor(&design_used, nrows, ncols, &beta0, &mut workspace.mu0);
        predict_colmajor(&design_used, nrows, ncols, &beta1, &mut workspace.mu1);
        let psi_target =
            if matches!(problem.target_population, TargetPopulation::CustomDistribution(_)) {
                TargetPopulation::AllObserved
            } else {
                problem.target_population.clone()
            };
        aipw_psi(
            &t_used,
            &y_used,
            &e_used,
            &workspace.mu0,
            &workspace.mu1,
            &psi_target,
            &mut workspace.psi,
        )?;
        let ate = if matches!(problem.target_population, TargetPopulation::CustomDistribution(_)) {
            crate::joint_if::weighted_mean(&workspace.psi, problem.target_weights.as_deref())?
        } else {
            workspace.psi.iter().sum::<f64>() / workspace.psi.len() as f64
        };
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
            center_population_psi(&mut workspace.psi, &t_used, &problem.target_population, ate);
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
            };
            correct_aipw_psi_for_nuisances(&mut workspace.psi, &fit)?;
        }
        let se_analytic = crate::se::influence_se_kind(
            self.se_kind,
            &workspace.psi,
            problem.nrows,
            self.cluster_ids.as_deref(),
            self.multiway_ids.as_deref(),
            self.panel_times.as_deref(),
            retained.as_deref(),
        )?;
        let boot = if self.bootstrap_replicates == 0 {
            None
        } else {
            Some(self.bootstrap_se(problem, workspace, ctx)?)
        };
        let overlap_report = Some(crate::propensity::propensity_overlap_report(
            problem,
            &model.fit.scores,
            None,
            IpwTarget::from_population(&problem.target_population).ok(),
        ));
        let estimate = EffectEstimate::new(ate, se_analytic, assumptions, problem.overlap)
            .with_se_kind(self.se_kind)
            .with_overlap_report(overlap_report)
            .with_bootstrap(boot)
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
            .target_weights
            .as_deref()
            .ok_or_else(|| EstimationError::data_msg("missing custom target weights"))?;
        self.fit_crossfit_scores(problem, ctx, assumptions, Some(weights), true)
    }

    /// Cross-fitted binary AIPW contrast. Uniform weights (`None`) are the
    /// `AllObserved` retarget(`1`) object.
    fn fit_crossfit_scores(
        &self,
        problem: &PreparedPropensityProblem,
        ctx: &ExecutionContext,
        assumptions: AssumptionSet,
        weights: Option<&[f64]>,
        refuse_overlap: bool,
    ) -> Result<EffectEstimate, EstimationError> {
        // The recorded cross-fit seed must govern the fold plan, not only the learners.
        let mut seeded = problem.clone();
        seeded.fold_seed = ctx.rng.master_seed();
        let problem = &seeded;
        let build = |p: &PreparedPropensityProblem| {
            crate::crossfit_aipw::build_binary_scores(
                p,
                p.treatment_id,
                &[None],
                crate::crossfit_aipw::DEFAULT_AIPW_FOLDS,
                &self.glm_options,
                self.backend,
            )
        };
        let table = build(problem)?;
        let summary = table.summarize(weights)?;
        let contrast = table.linear_contrast(&summary, &[-1.0, 1.0])?;
        let iid_se = matches!(self.se_kind, AnalyticSeKind::Homoskedastic);
        let n = table.n_rows as f64;
        let influence: Vec<f64> = match weights {
            Some(w) => {
                let mass: f64 = w.iter().sum();
                table
                    .column(0)?
                    .iter()
                    .zip(table.column(1)?)
                    .zip(w)
                    .map(|((&a, &b), &wi)| n * wi / mass * (b - a - contrast.value))
                    .collect()
            }
            None => table.column(0)?.iter().zip(table.column(1)?).map(|(&a, &b)| b - a).collect(),
        };
        let boot = if self.bootstrap_replicates == 0 {
            None
        } else {
            Some(bootstrap_se(self.bootstrap_replicates, ctx, 0xA1D5, problem.nrows, |idx| {
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
                p.treatment = gather(&problem.treatment, idx).into();
                p.outcome = gather(&problem.outcome, idx).into();
                // Copies of one original row keep that row's identity (and shared fold), so
                // no unit is both in a training set and in its own validation fold.
                p.row_index = idx.iter().map(|&i| problem.row_index[i]).collect::<Vec<_>>().into();
                p.fold_assignment = problem
                    .fold_assignment
                    .as_deref()
                    .map(|ids| idx.iter().map(|&i| ids[i]).collect::<Vec<_>>().into());
                if let Some(w) = weights {
                    p.target_weights = Some(gather(w, idx).into());
                }
                let Ok(t) = build(&p) else {
                    return Ok(None);
                };
                let ss = t.summarize(p.target_weights.as_deref())?;
                Ok(Some(ss.means[1] - ss.means[0]))
            })?)
        };
        // The score table's covariance and simultaneous bands are iid objects; under a
        // dependence-robust SE kind they would contradict `se_analytic`, so they are withheld
        // rather than published at the wrong strength.
        let inference = table.inference(weights)?;
        if refuse_overlap && !inference.support.overlap_ok {
            return Err(EstimationError::unsupported("custom target weighted overlap failed"));
        }
        let se_analytic = if iid_se {
            contrast.se
        } else {
            crate::se::influence_se_kind(
                self.se_kind,
                &influence,
                problem.nrows,
                self.cluster_ids.as_deref(),
                self.multiway_ids.as_deref(),
                self.panel_times.as_deref(),
                None,
            )?
        };
        let e_hat =
            table.columns.iter().position(|c| c.arm == 1 && c.threshold.is_none()).and_then(
                |col| table.propensities.get(col * table.n_rows..(col + 1) * table.n_rows),
            );
        let overlap_report = e_hat.map(|scores| {
            crate::propensity::propensity_overlap_report(
                problem,
                scores,
                None,
                IpwTarget::from_population(&problem.target_population).ok(),
            )
        });
        let mut result =
            EffectEstimate::new(contrast.value, se_analytic, assumptions, problem.overlap)
                .with_se_kind(self.se_kind)
                .with_overlap_report(overlap_report)
                .with_joint_covariance(iid_se.then_some(summary.covariance))
                .with_bootstrap(boot)
                .with_influence(Some(influence.into()))
                .with_score_table(Some(table));
        result.score_inference = iid_se.then_some(inference);
        Ok(result)
    }

    fn bootstrap_se(
        &self,
        problem: &PreparedPropensityProblem,
        workspace: &mut AipwWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<BootstrapSeResult, EstimationError> {
        let clip = clip_of(problem.overlap);
        let trim = trim_of(problem.overlap);
        let n = problem.nrows;
        let ncols = problem.design_ncols;
        let _ = workspace;
        crate::util::bootstrap_se_with_scratch(
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
                crate::util::gather_bootstrap_vector(t_boot, &problem.treatment, idx);
                crate::util::gather_bootstrap_vector(y_boot, &problem.outcome, idx);
                crate::util::gather_bootstrap_design(x_boot, &problem.design_matrix, n, ncols, idx);
                if antecedent_stats::fit_propensity_in_place(
                    x_boot,
                    n,
                    ncols,
                    t_boot,
                    &self.backend,
                    &mut workspace.propensity,
                    &self.glm_options,
                )
                .is_err()
                {
                    return Ok(None);
                }
                let raw = &workspace.propensity.scores[..n];
                e.copy_from_slice(raw);
                if let Some(c) = clip {
                    clamp_scores(e, c);
                }
                let Ok(retained) = trim_retained_rows(raw, trim) else {
                    return Ok(None);
                };
                let (design_used, t_used, y_used, e_used): (&[f64], &[f64], &[f64], &[f64]) =
                    match &retained {
                        Some(rows) => {
                            select_rows_colmajor(x_boot, n, ncols, rows, design_trim);
                            gather_into(t_trim, t_boot, rows);
                            gather_into(y_trim, y_boot, rows);
                            gather_into(e_trim, e, rows);
                            (design_trim, t_trim, y_trim, e_trim)
                        }
                        None => (x_boot, t_boot, y_boot, e),
                    };
                let nrows = t_used.len();
                let Ok((beta0, beta1)) = fit_outcome_models(
                    design_used,
                    nrows,
                    ncols,
                    t_used,
                    y_used,
                    self.backend,
                    workspace,
                ) else {
                    return Ok(None);
                };
                predict_colmajor(design_used, nrows, ncols, &beta0, &mut workspace.mu0);
                predict_colmajor(design_used, nrows, ncols, &beta1, &mut workspace.mu1);
                if aipw_psi(
                    t_used,
                    y_used,
                    e_used,
                    &workspace.mu0,
                    &workspace.mu1,
                    &problem.target_population,
                    &mut workspace.psi,
                )
                .is_err()
                {
                    return Ok(None);
                }
                let m = workspace.psi.len() as f64;
                Ok(Some(workspace.psi.iter().sum::<f64>() / m))
            },
        )
    }
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
    let m = idx.len();
    out.clear();
    out.resize(m * ncols, 0.0);
    for c in 0..ncols {
        let src_base = c * nrows;
        let dst_base = c * m;
        for (r, &i) in idx.iter().enumerate() {
            out[dst_base + r] = matrix[src_base + i];
        }
    }
}

fn select_values(values: &[f64], idx: &[usize], out: &mut Vec<f64>) {
    out.clear();
    out.extend(idx.iter().map(|&i| values[i]));
}

/// Fit separate OLS outcome models on the control (`T=0`) and treated (`T=1`) arms of
/// `design_matrix` (column-major `[1 | Z…]`), returning `(beta0, beta1)`.
///
/// # Errors
///
/// Empty treated/control arm, or an OLS backend failure (e.g. rank deficiency within an arm).
pub(crate) fn fit_outcome_models(
    design_matrix: &[f64],
    nrows: usize,
    ncols: usize,
    treatment: &[f64],
    outcome: &[f64],
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
    let fit0 = backend
        .least_squares(
            &workspace.control_design,
            control_idx.len(),
            ncols,
            &workspace.control_outcome,
            &mut workspace.outcome,
        )
        .map_err(stats_err)?;
    let beta0 = fit0.coefficients;

    select_rows_colmajor(design_matrix, nrows, ncols, &treated_idx, &mut workspace.treated_design);
    select_values(outcome, &treated_idx, &mut workspace.treated_outcome);
    let fit1 = backend
        .least_squares(
            &workspace.treated_design,
            treated_idx.len(),
            ncols,
            &workspace.treated_outcome,
            &mut workspace.outcome,
        )
        .map_err(stats_err)?;
    let beta1 = fit1.coefficients;

    Ok((beta0, beta1))
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
fn aipw_psi(
    treatment: &[f64],
    outcome: &[f64],
    propensity: &[f64],
    mu0: &[f64],
    mu1: &[f64],
    target: &TargetPopulation,
    out: &mut Vec<f64>,
) -> Result<(), EstimationError> {
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
            let pi = treatment.iter().filter(|&&t| t > 0.5).count() as f64 / treatment.len() as f64;
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
            let pi0 =
                treatment.iter().filter(|&&t| t <= 0.5).count() as f64 / treatment.len() as f64;
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
) {
    let n = treatment.len() as f64;
    let arm_indicator =
        |t: f64, treated: bool| -> f64 { if (t > 0.5) == treated { 1.0 } else { 0.0 } };
    let treated = match target {
        TargetPopulation::Treated => true,
        TargetPopulation::Untreated => false,
        _ => return,
    };
    let share = treatment.iter().map(|&t| arm_indicator(t, treated)).sum::<f64>() / n;
    if share <= 0.0 {
        return;
    }
    for (value, &t) in psi.iter_mut().zip(treatment) {
        *value -= estimate * arm_indicator(t, treated) / share;
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
        #[allow(clippy::float_cmp)]
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
}

#[cfg(test)]
#[allow(clippy::many_single_char_names, clippy::float_cmp)]
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
            let p = 1.0 / (1.0 + (-logit).exp());
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

    fn ctx() -> ExecutionContext {
        ExecutionContext::for_tests(7)
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
        assert!(effect.se_bootstrap.is_some());
        assert!(effect.overlap_report.is_some());
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
            let pi_i = 1.0 / (1.0 + (-logit).exp());
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
            let pi_i = 1.0 / (1.0 + (-logit).exp());
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
            let pi_i = 1.0 / (1.0 + (-logit).exp());
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
            let pi_i = 1.0 / (1.0 + (-logit).exp());
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
            assert!(fit.se_analytic.is_finite() && fit.se_analytic > 0.0, "kind={kind:?}");
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
        let report = clean.overlap_report.as_ref().unwrap();
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
            let p = 1.0 / (1.0 + (-logit).exp());
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
        let rel = (fit.se_analytic - jk).abs() / jk;
        assert!(rel < 0.15, "analytic {} vs jackknife {jk}", fit.se_analytic);
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
            assert!(fit.score_table.is_some());
            assert!(fit.joint_covariance.is_none() && fit.score_inference.is_none());
        }
    }
}
