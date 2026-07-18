//! Inverse-probability weighting estimator.
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
use crate::util::{bootstrap_se, BootstrapSeResult};

/// Inverse-probability weighting estimator (ATE/ATT/ATC via `TargetPopulation`).
///
/// Point estimate is the Hajek (self-normalized) weighted difference of means. Positivity is
/// mandatory: [`OverlapPolicy::ExplicitOverride`] is refused.
#[derive(Clone, Debug)]
pub struct PropensityWeighting {
    /// Dense linear-algebra backend used for the logistic IRLS fit.
    pub backend: FaerBackend,
    /// Bootstrap replicates (0 = skip bootstrap).
    pub bootstrap_replicates: u32,
    /// Overlap policy; must be [`OverlapPolicy::RequireDiagnostics`].
    pub overlap: OverlapPolicy,
    /// GLM fitting options for the propensity model.
    pub glm_options: GlmOptions,
}

impl Default for PropensityWeighting {
    fn default() -> Self {
        Self::new()
    }
}

impl PropensityWeighting {
    /// Defaults: 200 bootstrap replicates, clip = 0.01, no trim.
    #[must_use]
    pub fn new() -> Self {
        Self {
            backend: FaerBackend,
            bootstrap_replicates: 200,
            overlap: default_propensity_overlap(),
            glm_options: GlmOptions::default(),
        }
    }

    /// Prepare the covariate design from tabular data, identified estimand, and query.
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
        prepare_propensity_problem(data, estimand, query, self.overlap)
    }

    /// Fit the propensity model and compute the Hajek-weighted effect, with optional bootstrap.
    ///
    /// # Errors
    ///
    /// Unsupported target population or GLM/backend failure.
    pub fn fit(
        &self,
        problem: &PreparedPropensityProblem,
        workspace: &mut PropensityEstimationWorkspace,
        ctx: &ExecutionContext,
        assumptions: AssumptionSet,
    ) -> Result<EffectEstimate, EstimationError> {
        let target = IpwTarget::from_population(&problem.target_population)?;
        let trim = trim_of(problem.overlap);
        let model = PropensityModel::fit(
            problem,
            &self.backend,
            &mut workspace.propensity,
            &self.glm_options,
        )?;

        let weights = compute_ipw_weights(
            &problem.treatment,
            &model.clipped_scores,
            &model.fit.scores,
            target,
            trim,
        );
        let ate = hajek_difference(&problem.treatment, &problem.outcome, &weights)?;
        let se_analytic = hajek_analytic_se(&problem.treatment, &problem.outcome, &weights);

        let boot = if self.bootstrap_replicates == 0 {
            None
        } else {
            Some(self.bootstrap_se(problem, target, trim, workspace, ctx)?)
        };

        let overlap_report = Some(OverlapReport::from_propensities(
            &model.fit.scores,
            Some(&weights),
            problem.overlap,
        ));

        Ok(EffectEstimate {
            ate,
            se_analytic,
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
        target: IpwTarget,
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
        bootstrap_se(self.bootstrap_replicates, ctx, 0x9A17_u64, n, |idx| {
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
            let mut clipped = raw.clone();
            if let Some(c) = clip {
                clamp_scores(&mut clipped, c);
            }
            let w = compute_ipw_weights(&t_boot, &clipped, &raw, target, trim);
            match hajek_difference(&t_boot, &y_boot, &w) {
                Ok(a) => Ok(Some(a)),
                Err(_) => Ok(None),
            }
        })
    }
}

// ---------------------------------------------------------------------------------------------
// Stratification
// ---------------------------------------------------------------------------------------------


// IPW weights + Hajek estimator (shared by `PropensityWeighting`)
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum IpwTarget {
    Ate,
    Att,
    Atc,
}

impl IpwTarget {
    fn from_population(pop: &TargetPopulation) -> Result<Self, EstimationError> {
        match pop {
            TargetPopulation::AllObserved => Ok(Self::Ate),
            TargetPopulation::Treated => Ok(Self::Att),
            TargetPopulation::Untreated => Ok(Self::Atc),
            _ => Err(EstimationError::unsupported("propensity weighting supports AllObserved, Treated, or Untreated target populations")),
        }
    }

    fn weight(self, t: f64, e: f64) -> f64 {
        match self {
            Self::Ate => {
                if t > 0.5 {
                    1.0 / e
                } else {
                    1.0 / (1.0 - e)
                }
            }
            Self::Att => {
                if t > 0.5 {
                    1.0
                } else {
                    e / (1.0 - e)
                }
            }
            Self::Atc => {
                if t > 0.5 {
                    (1.0 - e) / e
                } else {
                    1.0
                }
            }
        }
    }
}

/// `scores_for_weight` feeds the weight formula (typically clipped); `scores_for_trim` feeds
/// the trim decision (typically the raw, pre-clip scores) — they may be the same slice.
pub(crate) fn compute_ipw_weights(
    treatment: &[f64],
    scores_for_weight: &[f64],
    scores_for_trim: &[f64],
    target: IpwTarget,
    trim: Option<f64>,
) -> Vec<f64> {
    treatment
        .iter()
        .zip(scores_for_weight)
        .zip(scores_for_trim)
        .map(|((&t, &e), &raw)| {
            if let Some(tr) = trim {
                if raw < tr || raw > 1.0 - tr {
                    return 0.0;
                }
            }
            target.weight(t, e)
        })
        .collect()
}

pub(crate) fn hajek_difference(
    treatment: &[f64],
    outcome: &[f64],
    weights: &[f64],
) -> Result<f64, EstimationError> {
    let (mut num1, mut den1, mut num0, mut den0) = (0.0, 0.0, 0.0, 0.0);
    for ((&t, &y), &w) in treatment.iter().zip(outcome).zip(weights) {
        if t > 0.5 {
            num1 += w * y;
            den1 += w;
        } else {
            num0 += w * y;
            den0 += w;
        }
    }
    if den1 <= 0.0 || den0 <= 0.0 {
        return Err(EstimationError::data_msg("IPW weighting left an arm with zero total weight (trimming/clipping removed all treated or all control units)"));
    }
    Ok(num1 / den1 - num0 / den0)
}

pub(crate) fn hajek_weighted_mean(
    treatment: &[f64],
    outcome: &[f64],
    weights: &[f64],
    want_treated: bool,
) -> f64 {
    let (mut num, mut den) = (0.0, 0.0);
    for i in 0..treatment.len() {
        if (treatment[i] > 0.5) == want_treated {
            num += weights[i] * outcome[i];
            den += weights[i];
        }
    }
    if den > 0.0 { num / den } else { f64::NAN }
}

pub(crate) fn hajek_group_variance(
    treatment: &[f64],
    outcome: &[f64],
    weights: &[f64],
    want_treated: bool,
    mu: f64,
) -> f64 {
    let (mut num, mut den) = (0.0, 0.0);
    for i in 0..treatment.len() {
        if (treatment[i] > 0.5) == want_treated {
            let w = weights[i];
            num += w * w * (outcome[i] - mu).powi(2);
            den += w;
        }
    }
    if den > 0.0 { num / (den * den) } else { f64::NAN }
}

/// Linearized (ratio-estimator) analytic SE of the Hajek ATE/ATT/ATC difference.
pub(crate) fn hajek_analytic_se(treatment: &[f64], outcome: &[f64], weights: &[f64]) -> f64 {
    let mu1 = hajek_weighted_mean(treatment, outcome, weights, true);
    let mu0 = hajek_weighted_mean(treatment, outcome, weights, false);
    let v1 = hajek_group_variance(treatment, outcome, weights, true, mu1);
    let v0 = hajek_group_variance(treatment, outcome, weights, false, mu0);
    (v1 + v0).sqrt()
}
