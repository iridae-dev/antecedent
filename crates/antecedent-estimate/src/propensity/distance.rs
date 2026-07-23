//! Covariate-distance matching estimator.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{AssumptionSet, AverageEffectQuery, ExecutionContext, PopulationRegistry};
use antecedent_data::TabularData;
use antecedent_expr::IdentifiedEstimand;
use antecedent_stats::{FaerBackend, GlmOptions, MatchingDistance, fit_propensity};

use super::matching::matching_contrast;
use super::prepare::{
    PreparedPropensityProblem, PropensityEstimationWorkspace, PropensityModel,
    default_propensity_overlap, prepare_propensity_problem_with_registry, restrict_to_rows,
    to_row_major, trim_of, trim_retained_rows,
};
use crate::adjustment::EffectEstimate;
use crate::error::EstimationError;
use crate::overlap::{OverlapPolicy, OverlapReport};
use crate::se::AnalyticSeKind;
use crate::util::{BootstrapSeResult, bootstrap_se};

/// Distance matching on raw adjustment covariates (Euclidean), not the propensity score.
///
/// A propensity model is still fit to populate mandatory positivity diagnostics
/// ([`EffectEstimate::overlap_report`]) and — when the overlap policy sets a trim threshold —
/// to restrict matching to common-support rows; it does not otherwise influence the
/// covariate-space matched contrast. Positivity is mandatory:
/// [`OverlapPolicy::ExplicitOverride`] is refused.
#[derive(Clone, Debug)]
pub struct DistanceMatching {
    /// Dense linear-algebra backend used for the diagnostic logistic fit.
    pub backend: FaerBackend,
    /// Bootstrap replicates (0 = skip bootstrap).
    pub bootstrap_replicates: u32,
    /// Overlap policy; must be [`OverlapPolicy::RequireDiagnostics`].
    pub overlap: OverlapPolicy,
    /// GLM fitting options for the diagnostic propensity model.
    pub glm_options: GlmOptions,
    /// Optional maximum Euclidean distance for an accepted match.
    pub caliper: Option<f64>,
    /// Analytic SE kind.
    pub se_kind: AnalyticSeKind,
    /// Optional cluster ids aligned to prepared complete-case rows.
    pub cluster_ids: Option<Vec<u32>>,
    /// Optional bindings for named predicates / custom target distributions.
    pub population_registry: Option<PopulationRegistry>,
    /// Multiway cluster ids (one `Vec<u32>` per clustering dimension).
    pub multiway_ids: Option<Vec<Vec<u32>>>,
}

impl Default for DistanceMatching {
    fn default() -> Self {
        Self::new()
    }
}

impl DistanceMatching {
    /// Defaults: no caliper, 200 bootstrap replicates, clip = 0.01, no trim.
    #[must_use]
    pub fn new() -> Self {
        Self {
            backend: FaerBackend,
            bootstrap_replicates: 200,
            overlap: default_propensity_overlap(),
            glm_options: GlmOptions::default(),
            caliper: None,
            se_kind: AnalyticSeKind::Homoskedastic,
            cluster_ids: None,
            population_registry: None,
            multiway_ids: None,
        }
    }

    /// Prepare the covariate design.
    ///
    /// # Errors
    ///
    /// See [`PropensityWeighting::prepare`].
    pub fn prepare(
        &self,
        data: &TabularData,
        estimand: &IdentifiedEstimand,
        query: &AverageEffectQuery,
    ) -> Result<PreparedPropensityProblem, EstimationError> {
        prepare_propensity_problem_with_registry(
            data,
            estimand,
            query,
            self.overlap,
            self.population_registry.as_ref(),
        )
    }

    /// Match on raw covariates (Euclidean) and compute the matched effect; fits a diagnostic
    /// propensity model for the mandatory overlap report.
    ///
    /// # Errors
    ///
    /// Empty adjustment set, unsupported target population, empty treated/control arm, no
    /// matches within the caliper, or GLM failure (diagnostic fit).
    pub fn fit(
        &self,
        problem: &PreparedPropensityProblem,
        workspace: &mut PropensityEstimationWorkspace,
        ctx: &ExecutionContext,
        assumptions: AssumptionSet,
    ) -> Result<EffectEstimate, EstimationError> {
        if problem.adjustment_set.is_empty() {
            return Err(EstimationError::unsupported(
                "distance matching requires a non-empty adjustment set",
            ));
        }
        let trim = trim_of(problem.overlap);
        let dim = problem.covariates.len();
        let features = to_row_major(&problem.covariates, problem.nrows);

        // Diagnostic propensity fit: populates the mandatory overlap report and, when a trim
        // threshold is configured, restricts both query and donor sets to the common-support
        // rows (raw scores) — it does not otherwise influence the covariate-space contrast.
        let diag = PropensityModel::fit(
            problem,
            &self.backend,
            &mut workspace.propensity,
            &self.glm_options,
        )?;
        let retained = trim_retained_rows(&diag.fit.scores, trim)?;
        let (t_used, y_used, f_used) = restrict_to_rows(
            &problem.treatment,
            &problem.outcome,
            &features,
            dim,
            retained.as_deref(),
        );
        let tw_used: Option<Vec<f64>> = problem.target_weights.as_ref().map(|w| match &retained {
            Some(idx) => idx.iter().map(|&i| w[i]).collect(),
            None => w.to_vec(),
        });
        let clusters_used = match (&self.cluster_ids, &retained) {
            (Some(ids), Some(idx)) => {
                if ids.len() != problem.nrows {
                    return Err(EstimationError::data_msg(format!(
                        "cluster_ids length {} != nrows {}",
                        ids.len(),
                        problem.nrows
                    )));
                }
                Some(idx.iter().map(|&i| ids[i]).collect::<Vec<_>>())
            }
            (Some(ids), None) => {
                if ids.len() != problem.nrows {
                    return Err(EstimationError::data_msg(format!(
                        "cluster_ids length {} != nrows {}",
                        ids.len(),
                        problem.nrows
                    )));
                }
                Some(ids.clone())
            }
            (None, _) => None,
        };
        let result = matching_contrast(
            &t_used,
            &y_used,
            &f_used,
            dim,
            MatchingDistance::Euclidean,
            &problem.target_population,
            self.caliper,
            workspace,
            self.se_kind,
            clusters_used.as_deref(),
            tw_used.as_deref(),
            self.multiway_ids.as_ref(),
        )?;

        let boot = if self.bootstrap_replicates == 0 {
            None
        } else {
            Some(self.bootstrap_se(problem, dim, &features, trim, workspace, ctx)?)
        };

        let overlap_report =
            Some(OverlapReport::from_propensities(&diag.fit.scores, None, problem.overlap));

        Ok(EffectEstimate {
            ate: result.ate,
            se_analytic: result.se_analytic,
            se_bootstrap: None,
            bootstrap_replicates_ok: None,
            bootstrap_replicates_failed: None,
            bootstrap_cancelled: false,
            bootstrap_early_stopped: false,
            assumptions,
            overlap: problem.overlap,
            overlap_report,
            retained_memory_bytes: Some(workspace.retained_memory_bytes()),
        }
        .with_bootstrap(boot))
    }

    fn bootstrap_se(
        &self,
        problem: &PreparedPropensityProblem,
        dim: usize,
        features: &[f64],
        trim: Option<f64>,
        workspace: &mut PropensityEstimationWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<BootstrapSeResult, EstimationError> {
        let n = problem.nrows;
        let ncols = problem.design_ncols;
        let mut feat_boot = vec![0.0; n * dim];
        // Diagnostic design resample, needed only to recompute the trim per replicate.
        let mut x_boot = if trim.is_some() { vec![0.0; n * ncols] } else { Vec::new() };
        let mut t_boot = vec![0.0; n];
        let mut y_boot = vec![0.0; n];
        bootstrap_se(self.bootstrap_replicates, ctx, 0x7C11_u64, n, |idx| {
            for (r, &src) in idx.iter().enumerate() {
                t_boot[r] = problem.treatment[src];
                y_boot[r] = problem.outcome[src];
                for d in 0..dim {
                    feat_boot[r * dim + d] = features[src * dim + d];
                }
                if trim.is_some() {
                    for c in 0..ncols {
                        x_boot[c * n + r] = problem.design_matrix[c * n + src];
                    }
                }
            }
            let retained = if trim.is_some() {
                let Ok(fit) = fit_propensity(
                    &x_boot,
                    n,
                    ncols,
                    &t_boot,
                    &self.backend,
                    &mut workspace.propensity,
                    &self.glm_options,
                ) else {
                    return Ok(None);
                };
                match trim_retained_rows(&fit.scores, trim) {
                    Ok(r) => r,
                    Err(_) => return Ok(None),
                }
            } else {
                None
            };
            let (t_used, y_used, f_used) =
                restrict_to_rows(&t_boot, &y_boot, &feat_boot, dim, retained.as_deref());
            match matching_contrast(
                &t_used,
                &y_used,
                &f_used,
                dim,
                MatchingDistance::Euclidean,
                &problem.target_population,
                self.caliper,
                workspace,
                AnalyticSeKind::Homoskedastic,
                None,
                None,
                None,
            ) {
                Ok(m) => Ok(Some(m.ate)),
                Err(_) => Ok(None),
            }
        })
    }
}
