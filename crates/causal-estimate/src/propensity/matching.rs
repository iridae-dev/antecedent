//! Propensity-score nearest-neighbor matching.
//!
//! Analytic standard errors follow Abadie–Imbens (2006) with donor-usage counts
//! `Kᵢ` (matching with replacement). A linear within-arm regression bias
//! adjustment (Abadie–Imbens) is applied on the match feature(s).
//!
//! **Bootstrap caution:** the nonparametric bootstrap is invalid for nearest-neighbor
//! matching with a fixed number of matches (Abadie–Imbens 2008). Prefer the analytic
//! SE; bootstrap replicates (when enabled) are retained only for diagnostics and must
//! not be treated as valid confidence-interval input for NN matching.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use causal_core::{AssumptionSet, AverageEffectQuery, ExecutionContext, TargetPopulation};
use causal_data::TabularData;
use causal_expr::IdentifiedEstimand;
use causal_stats::{
    FaerBackend, GlmOptions, MatchingDistance, fit_propensity,
};

use super::prepare::{
    PreparedPropensityProblem, PropensityEstimationWorkspace, PropensityModel, clamp_scores,
    clip_of, default_propensity_overlap, gather, gather_rowmajor, prepare_propensity_problem,
    restrict_to_rows, split_by_treatment, trim_of, trim_retained_rows,
};
use crate::adjustment::EffectEstimate;
use crate::error::EstimationError;
use crate::overlap::{OverlapPolicy, OverlapReport};
use crate::se::{AnalyticSeKind, cluster_influence_se};
use crate::util::{bootstrap_se, sample_std, stats_err, BootstrapSeResult};

/// Propensity-score nearest-neighbor matching (Absolute distance, optional caliper).
///
/// Positivity is mandatory: [`OverlapPolicy::ExplicitOverride`] is refused. Supports
/// ATT/ATC/ATE via `TargetPopulation`.
///
/// Analytic SEs use Abadie–Imbens (2006) donor-reuse variance; see module docs for the
/// bootstrap caveat (Abadie–Imbens 2008).
#[derive(Clone, Debug)]
pub struct PropensityMatching {
    /// Dense linear-algebra backend used for the logistic IRLS fit.
    pub backend: FaerBackend,
    /// Bootstrap replicates (0 = skip bootstrap). Invalid for NN matching CIs — see module docs.
    pub bootstrap_replicates: u32,
    /// Overlap policy; must be [`OverlapPolicy::RequireDiagnostics`].
    pub overlap: OverlapPolicy,
    /// GLM fitting options for the propensity model.
    pub glm_options: GlmOptions,
    /// Optional maximum propensity distance for an accepted match.
    pub caliper: Option<f64>,
    /// Analytic SE kind (Abadie–Imbens / hetero / cluster).
    pub se_kind: AnalyticSeKind,
    /// Optional cluster ids aligned to prepared complete-case rows.
    pub cluster_ids: Option<Vec<u32>>,
}

impl Default for PropensityMatching {
    fn default() -> Self {
        Self::new()
    }
}

impl PropensityMatching {
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

    /// Fit the propensity model and compute the matched effect.
    ///
    /// # Errors
    ///
    /// Unsupported target population, empty treated/control arm, no matches within the
    /// caliper, or GLM failure.
    pub fn fit(
        &self,
        problem: &PreparedPropensityProblem,
        workspace: &mut PropensityEstimationWorkspace,
        ctx: &ExecutionContext,
        assumptions: AssumptionSet,
    ) -> Result<EffectEstimate, EstimationError> {
        let trim = trim_of(problem.overlap);
        let model = PropensityModel::fit(
            problem,
            &self.backend,
            &mut workspace.propensity,
            &self.glm_options,
        )?;
        // Trim on RAW scores (mirrors PropensityWeighting): both query and donor sets are
        // restricted to common-support rows before matching.
        let retained = trim_retained_rows(&model.fit.scores, trim)?;
        let (t_used, y_used, s_used) = restrict_to_rows(
            &problem.treatment,
            &problem.outcome,
            &model.clipped_scores,
            1,
            retained.as_deref(),
        );
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
            &s_used,
            1,
            MatchingDistance::Absolute,
            &problem.target_population,
            self.caliper,
            workspace,
            self.se_kind,
            clusters_used.as_deref(),
        )?;

        let boot = if self.bootstrap_replicates == 0 {
            None
        } else {
            Some(self.bootstrap_se(problem, trim, workspace, ctx)?)
        };

        let overlap_report =
            Some(OverlapReport::from_propensities(&model.fit.scores, None, problem.overlap));

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
        bootstrap_se(self.bootstrap_replicates, ctx, 0x51E7_u64, n, |idx| {
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
            match matching_contrast(
                &t_used,
                &y_used,
                &s_used,
                1,
                MatchingDistance::Absolute,
                &problem.target_population,
                self.caliper,
                workspace,
                AnalyticSeKind::Homoskedastic,
                None,
            ) {
                Ok(m) => Ok(Some(m.ate)),
                Err(_) => Ok(None),
            }
        })
    }
}

/// Match each `query` row to its nearest `donor` row; returns bias-corrected
/// `query_y[q] − donor_y[matched]` and the local donor indices used (for `Kᵢ`).
///
/// Reuses [`PropensityEstimationWorkspace`]'s cached [`MatchingIndex`] when donor geometry
/// is unchanged.
pub(crate) fn match_diffs(
    donor_features: &[f64],
    donor_outcome: &[f64],
    dim: usize,
    distance: MatchingDistance,
    query_features: &[f64],
    query_outcome: &[f64],
    caliper: Option<f64>,
    workspace: &mut PropensityEstimationWorkspace,
) -> Result<(Vec<f64>, Vec<usize>, Vec<usize>), EstimationError> {
    let n_donors = donor_outcome.len();
    if n_donors == 0 {
        return Err(EstimationError::data_msg("matching requires at least one donor row"));
    }
    workspace.ensure_matching_index(donor_features, dim, distance)?;
    let n_queries = query_outcome.len();
    let mut donor_rows = std::mem::take(&mut workspace.matching_donor_rows);
    let mut distances = std::mem::take(&mut workspace.matching_distances);
    donor_rows.clear();
    donor_rows.resize(n_queries, 0);
    distances.clear();
    distances.resize(n_queries, 0.0);
    {
        let index = workspace.matching_index.as_ref().expect("ensured");
        index
            .match_all(query_features, n_queries, caliper, &mut donor_rows, &mut distances)
            .map_err(stats_err)?;
    }
    let mut diffs = Vec::with_capacity(n_queries);
    let mut used_donors = Vec::with_capacity(n_queries);
    let mut used_queries = Vec::with_capacity(n_queries);
    let mu_donor = fit_linear_mean(donor_features, donor_outcome, dim);
    for q in 0..n_queries {
        let d = donor_rows[q];
        if d != usize::MAX {
            let raw = query_outcome[q] - donor_outcome[d];
            let bias = match &mu_donor {
                Some(beta) => {
                    let mq = predict_linear(beta, query_features, dim, q);
                    let md = predict_linear(beta, donor_features, dim, d);
                    mq - md
                }
                None => 0.0,
            };
            diffs.push(raw - bias);
            used_donors.push(d);
            used_queries.push(q);
        }
    }
    workspace.matching_donor_rows = donor_rows;
    workspace.matching_distances = distances;
    Ok((diffs, used_donors, used_queries))
}

pub(crate) struct MatchedEstimate {
    pub(crate) ate: f64,
    pub(crate) se_analytic: f64,
}

/// ATT/ATC/ATE via nearest-neighbor matching on `features` (dim columns, row-major).
///
/// ATT matches treated→nearest control; ATC matches control→nearest treated (sign-flipped);
/// ATE pools both directions' per-unit imputed effects (Abadie–Imbens style).
#[allow(clippy::too_many_arguments)]
pub(crate) fn matching_contrast(
    treatment: &[f64],
    outcome: &[f64],
    features: &[f64],
    dim: usize,
    distance: MatchingDistance,
    target: &TargetPopulation,
    caliper: Option<f64>,
    workspace: &mut PropensityEstimationWorkspace,
    se_kind: AnalyticSeKind,
    cluster_ids: Option<&[u32]>,
) -> Result<MatchedEstimate, EstimationError> {
    if let Some(ids) = cluster_ids {
        if ids.len() != treatment.len() {
            return Err(EstimationError::data_msg(
                "matching cluster_ids length != treatment rows",
            ));
        }
    }
    let (treated_idx, control_idx) = split_by_treatment(treatment);
    if treated_idx.is_empty() || control_idx.is_empty() {
        return Err(EstimationError::data_msg("matching requires both treated and control rows"));
    }
    let treated_feat = gather_rowmajor(features, dim, &treated_idx);
    let control_feat = gather_rowmajor(features, dim, &control_idx);
    let treated_y = gather(outcome, &treated_idx);
    let control_y = gather(outcome, &control_idx);

    let (per_unit_effects, donor_usage, n_donors, effect_rows): (
        Vec<f64>,
        Vec<usize>,
        usize,
        Vec<usize>,
    ) = match target {
        TargetPopulation::Treated => {
            let (diffs, donors, q_local) = match_diffs(
                &control_feat,
                &control_y,
                dim,
                distance,
                &treated_feat,
                &treated_y,
                caliper,
                workspace,
            )?;
            let rows: Vec<usize> = q_local.iter().map(|&q| treated_idx[q]).collect();
            (diffs, donors, control_y.len(), rows)
        }
        TargetPopulation::Untreated => {
            let (diffs, donors, q_local) = match_diffs(
                &treated_feat,
                &treated_y,
                dim,
                distance,
                &control_feat,
                &control_y,
                caliper,
                workspace,
            )?;
            let flipped: Vec<f64> = diffs.into_iter().map(|d| -d).collect();
            let rows: Vec<usize> = q_local.iter().map(|&q| control_idx[q]).collect();
            (flipped, donors, treated_y.len(), rows)
        }
        TargetPopulation::AllObserved => {
            let (att_diffs, att_donors, att_q) = match_diffs(
                &control_feat,
                &control_y,
                dim,
                distance,
                &treated_feat,
                &treated_y,
                caliper,
                workspace,
            )?;
            let (atc_raw, atc_donors, atc_q) = match_diffs(
                &treated_feat,
                &treated_y,
                dim,
                distance,
                &control_feat,
                &control_y,
                caliper,
                workspace,
            )?;
            let atc_diffs: Vec<f64> = atc_raw.into_iter().map(|d| -d).collect();
            let n_control = control_y.len();
            let mut effects = att_diffs;
            effects.extend(atc_diffs);
            let mut donors = att_donors;
            donors.extend(atc_donors.into_iter().map(|d| d + n_control));
            let mut rows: Vec<usize> = att_q.iter().map(|&q| treated_idx[q]).collect();
            rows.extend(atc_q.iter().map(|&q| control_idx[q]));
            (effects, donors, n_control + treated_y.len(), rows)
        }
        _ => {
            return Err(EstimationError::UnsupportedQuery(
                "matching estimators support AllObserved, Treated, or Untreated target populations"
                    .into(),
            ));
        }
    };
    if per_unit_effects.is_empty() {
        return Err(EstimationError::data_msg("no matched units within caliper"));
    }
    let ate = per_unit_effects.iter().sum::<f64>() / per_unit_effects.len() as f64;
    let se_analytic = match se_kind {
        AnalyticSeKind::Homoskedastic => {
            abadie_imbens_se(&per_unit_effects, &donor_usage, n_donors)
        }
        AnalyticSeKind::Hc0
        | AnalyticSeKind::Hc1
        | AnalyticSeKind::Hc2
        | AnalyticSeKind::Hc3
        | AnalyticSeKind::NeweyWest { .. } => {
            abadie_imbens_se_hetero(&per_unit_effects, &donor_usage, n_donors)
        }
        AnalyticSeKind::Cluster | AnalyticSeKind::PanelClusterHac { .. } => {
            let Some(ids) = cluster_ids else {
                return Err(EstimationError::UnsupportedQuery(
                    "AnalyticSeKind::Cluster requires matching cluster_ids".into(),
                ));
            };
            let groups: Vec<u32> = effect_rows.iter().map(|&r| ids[r]).collect();
            cluster_influence_se(&per_unit_effects, &groups)
        }
        AnalyticSeKind::Multiway => {
            return Err(EstimationError::UnsupportedQuery(
                "matching AnalyticSeKind::Multiway is not supported; use Cluster or bootstrap"
                    .into(),
            ));
        }
    };
    Ok(MatchedEstimate { ate, se_analytic })
}

/// Abadie–Imbens (2006) SE for 1-NN matching with replacement (homoskedastic).
///
/// With unit-level matched effects `τ̂ᵢ` and donor reuse counts `Kⱼ`,
/// `Var = σ̂² (n + Σⱼ Kⱼ²) / n²` where `σ̂² = Var(τ̂ᵢ) / 2` (equal-arm residual variance).
fn abadie_imbens_se(effects: &[f64], donor_local: &[usize], n_donors: usize) -> f64 {
    let n = effects.len();
    if n < 2 || donor_local.len() != n {
        return sample_std(effects) / (n as f64).sqrt();
    }
    let mut k = vec![0usize; n_donors.max(1)];
    for &d in donor_local {
        if d < k.len() {
            k[d] += 1;
        }
    }
    let mean = effects.iter().sum::<f64>() / n as f64;
    let var_tau =
        effects.iter().map(|e| (e - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0);
    let sigma2 = (var_tau * 0.5).max(0.0);
    let sum_k2: f64 = k.iter().map(|&kj| (kj as f64).powi(2)).sum();
    let var = sigma2 * (n as f64 + sum_k2) / (n as f64).powi(2);
    var.sqrt()
}

/// Heteroskedastic Abadie–Imbens SE using pair-level variance proxies.
fn abadie_imbens_se_hetero(effects: &[f64], donor_local: &[usize], n_donors: usize) -> f64 {
    let n = effects.len();
    if n < 2 || donor_local.len() != n {
        return sample_std(effects) / (n as f64).sqrt();
    }
    let mut k = vec![0usize; n_donors.max(1)];
    for &d in donor_local {
        if d < k.len() {
            k[d] += 1;
        }
    }
    let mut var = 0.0;
    for (i, &d) in donor_local.iter().enumerate() {
        let sigma2_i = 0.5 * effects[i] * effects[i];
        let kd = k.get(d).copied().unwrap_or(0) as f64;
        var += sigma2_i * (1.0 + kd).powi(2);
    }
    (var / (n as f64).powi(2)).max(0.0).sqrt()
}

/// OLS of `y` on `[1, x]` (row-major `x` with `dim` columns). Returns `[intercept, β…]`.
fn fit_linear_mean(features: &[f64], y: &[f64], dim: usize) -> Option<Vec<f64>> {
    let n = y.len();
    if n < dim + 1 || dim == 0 {
        // Fall back to intercept-only mean when underdetermined.
        if n == 0 {
            return None;
        }
        return Some(vec![y.iter().sum::<f64>() / n as f64]);
    }
    let p = dim + 1;
    let mut xtx = vec![0.0; p * p];
    let mut xty = vec![0.0; p];
    for i in 0..n {
        let mut row = vec![1.0; p];
        for d in 0..dim {
            row[d + 1] = features[i * dim + d];
        }
        for a in 0..p {
            xty[a] += row[a] * y[i];
            for b in 0..p {
                xtx[a * p + b] += row[a] * row[b];
            }
        }
    }
    solve_linear_system(&mut xtx, &mut xty, p)
}

fn predict_linear(beta: &[f64], features: &[f64], dim: usize, row: usize) -> f64 {
    if beta.len() == 1 {
        return beta[0];
    }
    let mut y = beta[0];
    for d in 0..dim.min(beta.len().saturating_sub(1)) {
        y += beta[d + 1] * features[row * dim + d];
    }
    y
}

/// Gaussian elimination with partial pivoting; returns solution in `b`, or `None` if singular.
fn solve_linear_system(a: &mut [f64], b: &mut [f64], p: usize) -> Option<Vec<f64>> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn abadie_imbens_se_grows_with_donor_reuse() {
        let effects = vec![1.0, 1.2, 0.8, 1.1];
        // Four queries, two unique donors reused twice each.
        let donors_reuse = vec![0usize, 0, 1, 1];
        let donors_unique = vec![0usize, 1, 2, 3];
        let se_reuse = abadie_imbens_se(&effects, &donors_reuse, 2);
        let se_unique = abadie_imbens_se(&effects, &donors_unique, 4);
        assert!(
            se_reuse > se_unique,
            "reuse={se_reuse} unique={se_unique}"
        );
    }
}
