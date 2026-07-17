//! Propensity stratification estimator.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use causal_core::{AssumptionSet, AverageEffectQuery, ExecutionContext, TargetPopulation};
use causal_data::TabularData;
use causal_expr::IdentifiedEstimand;
use causal_stats::{FaerBackend, GlmOptions, PropensityWorkspace, fit_propensity};

use super::prepare::{
    PreparedPropensityProblem, PropensityEstimationWorkspace, PropensityModel, clamp_scores,
    clip_of, default_propensity_overlap, prepare_propensity_problem, restrict_to_rows, trim_of,
    trim_retained_rows,
};
use crate::adjustment::EffectEstimate;
use crate::error::EstimationError;
use crate::overlap::{OverlapPolicy, OverlapReport};
use crate::util::{bootstrap_se, sample_std, BootstrapSeResult};

/// Propensity stratification estimator: within-stratum difference of means pooled by size.
///
/// Supports [`TargetPopulation::AllObserved`] only (stratified ATE). Positivity is mandatory:
/// [`OverlapPolicy::ExplicitOverride`] is refused.
#[derive(Clone, Debug)]
pub struct PropensityStratification {
    /// Dense linear-algebra backend used for the logistic IRLS fit.
    pub backend: FaerBackend,
    /// Bootstrap replicates (0 = skip bootstrap).
    pub bootstrap_replicates: u32,
    /// Overlap policy; must be [`OverlapPolicy::RequireDiagnostics`].
    pub overlap: OverlapPolicy,
    /// GLM fitting options for the propensity model.
    pub glm_options: GlmOptions,
    /// Number of quantile strata (default 5).
    pub n_strata: u32,
}

impl Default for PropensityStratification {
    fn default() -> Self {
        Self::new()
    }
}

impl PropensityStratification {
    /// Defaults: 5 strata, 200 bootstrap replicates, clip = 0.01, no trim.
    #[must_use]
    pub fn new() -> Self {
        Self {
            backend: FaerBackend,
            bootstrap_replicates: 200,
            overlap: default_propensity_overlap(),
            glm_options: {
                let mut o = GlmOptions::default();
                o.ridge_on_separation = Some(crate::se::DEFAULT_RIDGE_ON_SEPARATION);
                o
            },
            n_strata: 5,
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
        prepare_propensity_problem(data, estimand, query, self.overlap)
    }

    /// Fit the propensity model, assign quantile strata, and compute the pooled stratified ATE.
    ///
    /// # Errors
    ///
    /// Unsupported target population, no strata with both arms represented, or GLM failure.
    pub fn fit(
        &self,
        problem: &PreparedPropensityProblem,
        workspace: &mut PropensityEstimationWorkspace,
        ctx: &ExecutionContext,
        assumptions: AssumptionSet,
    ) -> Result<EffectEstimate, EstimationError> {
        if !matches!(problem.target_population, TargetPopulation::AllObserved) {
            return Err(EstimationError::UnsupportedQuery(
                "propensity stratification only supports TargetPopulation::AllObserved".into(),
            ));
        }
        let trim = trim_of(problem.overlap);
        let model = PropensityModel::fit(
            problem,
            &self.backend,
            &mut workspace.propensity,
            &self.glm_options,
        )?;
        let n_strata = (self.n_strata.max(1)) as usize;
        // Trim on RAW scores (mirrors PropensityWeighting): stratify only common-support rows.
        let retained = trim_retained_rows(&model.fit.scores, trim)?;
        let (t_used, y_used, s_used) = restrict_to_rows(
            &problem.treatment,
            &problem.outcome,
            &model.clipped_scores,
            1,
            retained.as_deref(),
        );
        let stratum = assign_strata(&s_used, n_strata);
        let result = stratified_ate(&t_used, &y_used, &stratum, n_strata)?;

        let boot = if self.bootstrap_replicates == 0 {
            None
        } else {
            Some(self.bootstrap_se(problem, n_strata, trim, workspace, ctx)?)
        };

        let mut report = OverlapReport::from_propensities(&model.fit.scores, None, problem.overlap);
        // Strata missing a treatment arm are dropped from the pooled contrast; fold the
        // retained fraction into the support figure so the artifact reflects the population
        // the estimate actually targets.
        report.target_population_support *= result.retained_fraction;
        let overlap_report = Some(report);

        Ok(EffectEstimate {
            ate: result.ate,
            se_analytic: result.se_analytic,
            se_bootstrap: None,
            bootstrap_replicates_ok: None,
            bootstrap_replicates_failed: None,
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
        n_strata: usize,
        trim: Option<f64>,
        workspace: &mut PropensityEstimationWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<BootstrapSeResult, EstimationError> {
        let clip = clip_of(problem.overlap);
                let n = problem.nrows;
        let ncols = problem.design_ncols;
        let mut x_boot = vec![0.0; n * ncols];
        let mut t_boot = vec![0.0; n];
        let mut y_boot = vec![0.0; n];
        bootstrap_se(self.bootstrap_replicates, ctx, 0x3D2F_u64, n, |idx| {
            for (r, &src) in idx.iter().enumerate() {
                t_boot[r] = problem.treatment[src];
                y_boot[r] = problem.outcome[src];
                for c in 0..ncols {
                    x_boot[c * n + r] = problem.design_matrix[c * n + src];
                }
            }
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
            let raw = fit.scores;
            let mut scores = raw.clone();
            if let Some(c) = clip {
                clamp_scores(&mut scores, c);
            }
            let Ok(retained) = trim_retained_rows(&raw, trim) else {
                return Ok(None);
            };
            let (t_used, y_used, s_used) =
                restrict_to_rows(&t_boot, &y_boot, &scores, 1, retained.as_deref());
            let stratum = assign_strata(&s_used, n_strata);
            match stratified_ate(&t_used, &y_used, &stratum, n_strata) {
                Ok(r) => Ok(Some(r.ate)),
                Err(_) => Ok(None),
            }
        })
    }
}

// ---------------------------------------------------------------------------------------------
// Matching (shared by PropensityMatching and DistanceMatching)
// ---------------------------------------------------------------------------------------------


pub(crate) fn assign_strata(scores: &[f64], n_strata: usize) -> Vec<usize> {
    let n = scores.len();
    let k = n_strata.max(1);
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| scores[a].partial_cmp(&scores[b]).unwrap_or(std::cmp::Ordering::Equal));
    let mut stratum = vec![0usize; n];
    for (rank, &orig) in order.iter().enumerate() {
        let s = (rank * k) / n.max(1);
        stratum[orig] = s.min(k - 1);
    }
    stratum
}

struct StratifiedResult {
    ate: f64,
    se_analytic: f64,
    /// Fraction of input rows that landed in a stratum with both arms represented. Strata
    /// missing an arm are dropped from the pooled contrast, which redefines the target
    /// population; callers surface this via the overlap report's support figure.
    retained_fraction: f64,
}

pub(crate) fn stratified_ate(
    treatment: &[f64],
    outcome: &[f64],
    stratum: &[usize],
    n_strata: usize,
) -> Result<StratifiedResult, EstimationError> {
    let mut sum1 = vec![0.0; n_strata];
    let mut sq1 = vec![0.0; n_strata];
    let mut cnt1 = vec![0usize; n_strata];
    let mut sum0 = vec![0.0; n_strata];
    let mut sq0 = vec![0.0; n_strata];
    let mut cnt0 = vec![0usize; n_strata];
    for i in 0..treatment.len() {
        let s = stratum[i];
        if treatment[i] > 0.5 {
            sum1[s] += outcome[i];
            sq1[s] += outcome[i] * outcome[i];
            cnt1[s] += 1;
        } else {
            sum0[s] += outcome[i];
            sq0[s] += outcome[i] * outcome[i];
            cnt0[s] += 1;
        }
    }
    let mut diffs = Vec::new();
    let mut ns = Vec::new();
    let mut vars = Vec::new();
    for s in 0..n_strata {
        if cnt1[s] == 0 || cnt0[s] == 0 {
            continue;
        }
        let n1 = cnt1[s] as f64;
        let n0 = cnt0[s] as f64;
        let mean1 = sum1[s] / n1;
        let mean0 = sum0[s] / n0;
        // Unbiased sample variances (÷(n−1)); undefined (NaN) for a single-unit arm, which
        // honestly propagates into `se_analytic` rather than understating it.
        let var1 = sample_variance_from_moments(sq1[s], mean1, cnt1[s]);
        let var0 = sample_variance_from_moments(sq0[s], mean0, cnt0[s]);
        diffs.push(mean1 - mean0);
        ns.push(n1 + n0);
        vars.push(var1 / n1 + var0 / n0);
    }
    let total_n: f64 = ns.iter().sum();
    if total_n <= 0.0 {
        return Err(EstimationError::data_msg("no strata contain both treated and control units"));
    }
    let ate = diffs.iter().zip(&ns).map(|(d, n)| d * n).sum::<f64>() / total_n;
    let se_var = vars.iter().zip(&ns).map(|(v, n)| v * (n / total_n).powi(2)).sum::<f64>();
    let retained_fraction = total_n / (treatment.len().max(1) as f64);
    Ok(StratifiedResult { ate, se_analytic: se_var.sqrt(), retained_fraction })
}

/// Unbiased sample variance from `Σy²`, the mean, and the count (`NaN` when `count < 2`).
fn sample_variance_from_moments(sum_sq: f64, mean: f64, count: usize) -> f64 {
    if count < 2 {
        return f64::NAN;
    }
    let n = count as f64;
    ((sum_sq - n * mean * mean) / (n - 1.0)).max(0.0)
}
