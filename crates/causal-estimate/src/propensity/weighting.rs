//! Inverse-probability weighting estimator.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::many_single_char_names)]

use causal_core::{
    AssumptionSet, AverageEffectQuery, ExecutionContext, PopulationRegistry, TargetPopulation,
};
use causal_data::TabularData;
use causal_expr::IdentifiedEstimand;
use causal_stats::{FaerBackend, GlmOptions, fit_propensity};

use super::prepare::{
    PreparedPropensityProblem, PropensityEstimationWorkspace, PropensityModel, clamp_scores,
    clip_of, default_propensity_overlap, prepare_propensity_problem_with_registry, trim_of,
};
use crate::adjustment::EffectEstimate;
use crate::error::EstimationError;
use crate::overlap::{OverlapPolicy, OverlapReport};
use crate::util::{BootstrapSeResult, bootstrap_se};

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
    /// Optional bindings for named predicates / custom target distributions.
    pub population_registry: Option<PopulationRegistry>,
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
            population_registry: None,
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
        prepare_propensity_problem_with_registry(
            data,
            estimand,
            query,
            self.overlap,
            self.population_registry.as_ref(),
        )
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
        if matches!(target, IpwTarget::Custom) && problem.target_weights.is_none() {
            return Err(EstimationError::unsupported(
                "CustomDistribution requires PopulationRegistry weights on the prepared problem",
            ));
        }
        let trim = trim_of(problem.overlap);
        let model = PropensityModel::fit(
            problem,
            &self.backend,
            &mut workspace.propensity,
            &self.glm_options,
        )?;

        let mut weights = compute_ipw_weights(
            &problem.treatment,
            &model.clipped_scores,
            &model.fit.scores,
            target,
            trim,
        );
        apply_target_weights(&mut weights, problem.target_weights.as_deref());
        let ate = hajek_difference(&problem.treatment, &problem.outcome, &weights)?;
        let se_analytic = hajek_influence_se(
            &problem.treatment,
            &problem.outcome,
            &weights,
            &model.clipped_scores,
            &problem.design_matrix,
            problem.design_ncols,
        );

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
        let tw = problem.target_weights.as_deref();
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
            let mut w = compute_ipw_weights(&t_boot, &clipped, &raw, target, trim);
            if let Some(full_tw) = tw {
                for (r, &src) in idx.iter().enumerate() {
                    w[r] *= full_tw[src];
                }
            }
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
pub(crate) enum IpwTarget {
    Ate,
    Att,
    Atc,
    /// ATE-style IPW reweighted by `CustomDistribution` observation weights.
    Custom,
}

impl IpwTarget {
    fn from_population(pop: &TargetPopulation) -> Result<Self, EstimationError> {
        match pop {
            TargetPopulation::AllObserved | TargetPopulation::Predicate(_) => Ok(Self::Ate),
            TargetPopulation::CustomDistribution(_) => Ok(Self::Custom),
            TargetPopulation::Treated => Ok(Self::Att),
            TargetPopulation::Untreated => Ok(Self::Atc),
            _ => Err(EstimationError::unsupported(
                "propensity weighting supports AllObserved, Treated, Untreated, Predicate, or CustomDistribution",
            )),
        }
    }

    fn weight(self, t: f64, e: f64) -> f64 {
        match self {
            Self::Ate | Self::Custom => {
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

fn apply_target_weights(weights: &mut [f64], target_weights: Option<&[f64]>) {
    let Some(tw) = target_weights else {
        return;
    };
    for (w, &t) in weights.iter_mut().zip(tw) {
        *w *= t;
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
        return Err(EstimationError::data_msg(
            "IPW weighting left an arm with zero total weight (trimming/clipping removed all treated or all control units)",
        ));
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

/// Hajek ATE/ATT/ATC analytic SE via linearized influence scores, with a
/// first-order correction for estimated logistic propensity scores.
///
/// Orthogonalizes the Hajek ratio IF against the propensity score scores
/// `x_i (T_i − e_i)` so the reported SE is not conditional on weights as fixed.
pub(crate) fn hajek_influence_se(
    treatment: &[f64],
    outcome: &[f64],
    weights: &[f64],
    propensity: &[f64],
    design_colmajor: &[f64],
    ncols: usize,
) -> f64 {
    let n = treatment.len();
    if n < 2 || weights.len() != n || propensity.len() != n {
        return f64::NAN;
    }
    let mu1 = hajek_weighted_mean(treatment, outcome, weights, true);
    let mu0 = hajek_weighted_mean(treatment, outcome, weights, false);
    let (mut sum_w1, mut sum_w0) = (0.0, 0.0);
    for i in 0..n {
        if treatment[i] > 0.5 {
            sum_w1 += weights[i];
        } else {
            sum_w0 += weights[i];
        }
    }
    if sum_w1 <= 0.0 || sum_w0 <= 0.0 {
        return f64::NAN;
    }
    let nf = n as f64;
    let mut psi = vec![0.0; n];
    for i in 0..n {
        let (w1, w0) = if treatment[i] > 0.5 { (weights[i], 0.0) } else { (0.0, weights[i]) };
        // Linearized Hajek ratio contributions (E[ψ]=0).
        psi[i] = nf * (w1 / sum_w1) * (outcome[i] - mu1) - nf * (w0 / sum_w0) * (outcome[i] - mu0);
    }

    // Propensity scores s_{i,c} = x_{ic} (T_i − e_i). Residualize ψ on the score space.
    if ncols > 0 && design_colmajor.len() >= n * ncols {
        let mut scores = vec![0.0; n * ncols];
        for i in 0..n {
            let resid = treatment[i] - propensity[i];
            for c in 0..ncols {
                scores[c * n + i] = design_colmajor[c * n + i] * resid;
            }
        }
        // Gram matrix G = S'S / n and g = S'ψ / n; solve G α = g; ψ ← ψ − S α.
        let mut gram = vec![0.0; ncols * ncols];
        let mut rhs = vec![0.0; ncols];
        for c in 0..ncols {
            for i in 0..n {
                rhs[c] += scores[c * n + i] * psi[i];
            }
            rhs[c] /= nf;
            for d in 0..ncols {
                let mut acc = 0.0;
                for i in 0..n {
                    acc += scores[c * n + i] * scores[d * n + i];
                }
                gram[c * ncols + d] = acc / nf;
            }
        }
        if let Some(alpha) = solve_symmetric_posdef(&mut gram, &mut rhs, ncols) {
            for i in 0..n {
                let mut adj = 0.0;
                for c in 0..ncols {
                    adj += scores[c * n + i] * alpha[c];
                }
                psi[i] -= adj;
            }
        }
    }

    let mean = psi.iter().sum::<f64>() / nf;
    let var = psi.iter().map(|p| (p - mean).powi(2)).sum::<f64>() / (nf - 1.0);
    // Finite-sample inflation for estimated propensity (p design columns).
    let df = (nf - ncols.max(1) as f64).max(1.0);
    (var / nf * (nf / df)).max(0.0).sqrt()
}

/// Gaussian elimination for a small dense system (propensity score projection).
fn solve_symmetric_posdef(a: &mut [f64], b: &mut [f64], p: usize) -> Option<Vec<f64>> {
    for col in 0..p {
        let mut pivot = col;
        let mut best = a[col * p + col].abs();
        for r in (col + 1)..p {
            let v = a[r * p + col].abs();
            if v > best {
                best = v;
                pivot = r;
            }
        }
        if best < 1e-14 {
            return None;
        }
        if pivot != col {
            for c in 0..p {
                a.swap(col * p + c, pivot * p + c);
            }
            b.swap(col, pivot);
        }
        let diag = a[col * p + col];
        for r in (col + 1)..p {
            let f = a[r * p + col] / diag;
            for c in col..p {
                a[r * p + c] -= f * a[col * p + c];
            }
            b[r] -= f * b[col];
        }
    }
    let mut x = vec![0.0; p];
    for i in (0..p).rev() {
        let mut s = b[i];
        for j in (i + 1)..p {
            s -= a[i * p + j] * x[j];
        }
        x[i] = s / a[i * p + i];
    }
    Some(x)
}

/// Legacy weights-as-fixed Hajek SE (kept for differential reference in docs).
#[allow(dead_code)]
pub(crate) fn hajek_analytic_se(treatment: &[f64], outcome: &[f64], weights: &[f64]) -> f64 {
    let mu1 = hajek_weighted_mean(treatment, outcome, weights, true);
    let mu0 = hajek_weighted_mean(treatment, outcome, weights, false);
    let v1 = hajek_group_variance(treatment, outcome, weights, true, mu1);
    let v0 = hajek_group_variance(treatment, outcome, weights, false, mu0);
    (v1 + v0).sqrt()
}
