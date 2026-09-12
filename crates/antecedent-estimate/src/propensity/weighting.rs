//! Inverse-probability weighting estimator.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::many_single_char_names)]

use antecedent_core::{AssumptionSet, AverageEffectQuery, ExecutionContext, PopulationRegistry};
use antecedent_data::TabularData;
use antecedent_expr::IdentifiedEstimand;
use antecedent_stats::{FaerBackend, GlmOptions};

use super::prepare::{
    PreparedPropensityProblem, PropensityEstimationWorkspace, PropensityModel, clamp_scores,
    clip_of, default_propensity_overlap, prepare_propensity_problem_with_registry, trim_of,
};
use crate::adjustment::EffectEstimate;
use crate::error::EstimationError;
use crate::overlap::{IpwTarget, OverlapPolicy};
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

    /// Set the dense linear-algebra backend used for the logistic IRLS fit.
    #[must_use]
    pub const fn with_backend(mut self, backend: FaerBackend) -> Self {
        self.backend = backend;
        self
    }

    /// Set the number of bootstrap replicates used for the bootstrap standard error.
    ///
    /// Defaults to 200. Set to `0` to skip bootstrapping and report only the analytic SE.
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

    /// Set the GLM fitting options for the propensity model.
    #[must_use]
    pub const fn with_glm_options(mut self, glm_options: GlmOptions) -> Self {
        self.glm_options = glm_options;
        self
    }

    /// Set bindings for named predicates / custom target distributions.
    #[must_use]
    pub fn with_population_registry(mut self, registry: PopulationRegistry) -> Self {
        self.population_registry = Some(registry);
        self
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
            Some(HajekPropensity {
                scores: &model.fit.scores,
                design: &problem.design_matrix,
                ncols: problem.design_ncols,
                target,
                clip: clip_of(problem.overlap),
            }),
        )?;

        let boot = if self.bootstrap_replicates == 0 {
            None
        } else {
            Some(self.bootstrap_se(problem, target, trim, workspace, ctx)?)
        };

        let overlap_report = Some(crate::propensity::propensity_overlap_report(
            problem,
            &model.fit.scores,
            Some(&weights),
            Some(target),
        ));

        Ok(EffectEstimate::new(ate, se_analytic, assumptions, problem.overlap)
            .with_overlap_report(overlap_report)
            .with_retained_memory_bytes(Some(workspace.retained_memory_bytes()))
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
        // Replicate-invariant scratch: scores stay in `workspace.propensity.scores`,
        // and the clipped / weight buffers are reused across replicates.
        let mut clipped = vec![0.0; n];
        let mut w = vec![0.0; n];
        let tw = problem.target_weights.as_deref();
        bootstrap_se(self.bootstrap_replicates, ctx, 0x9A17_u64, n, |idx| {
            crate::util::gather_bootstrap_vector(&mut t_boot, &problem.treatment, idx);
            crate::util::gather_bootstrap_vector(&mut y_boot, &problem.outcome, idx);
            crate::util::gather_bootstrap_design(
                &mut x_boot,
                &problem.design_matrix,
                n,
                ncols,
                idx,
            );
            if antecedent_stats::fit_propensity_in_place(
                &x_boot,
                n,
                ncols,
                &t_boot,
                &self.backend,
                &mut workspace.propensity,
                &self.glm_options,
            )
            .is_err()
            {
                return Ok(None);
            }
            let raw = &workspace.propensity.scores[..n];
            clipped.copy_from_slice(raw);
            if let Some(c) = clip {
                clamp_scores(&mut clipped, c);
            }
            compute_ipw_weights_into(&mut w, &t_boot, &clipped, raw, target, trim);
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

// IPW weights + Hajek estimator (shared by `PropensityWeighting`)
// ---------------------------------------------------------------------------------------------

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
    let mut out = vec![0.0; treatment.len()];
    compute_ipw_weights_into(&mut out, treatment, scores_for_weight, scores_for_trim, target, trim);
    out
}

/// In-place form of [`compute_ipw_weights`] for replicate loops with a reused buffer.
pub(crate) fn compute_ipw_weights_into(
    out: &mut [f64],
    treatment: &[f64],
    scores_for_weight: &[f64],
    scores_for_trim: &[f64],
    target: IpwTarget,
    trim: Option<f64>,
) {
    for (slot, ((&t, &e), &raw)) in
        out.iter_mut().zip(treatment.iter().zip(scores_for_weight).zip(scores_for_trim))
    {
        *slot = if let Some(tr) = trim {
            if raw < tr || raw > 1.0 - tr { 0.0 } else { target.weight(t, e) }
        } else {
            target.weight(t, e)
        };
    }
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

/// Logistic nuisance fit and the weighting rule differentiated by the sandwich.
pub(crate) struct HajekPropensity<'a> {
    scores: &'a [f64],
    design: &'a [f64],
    ncols: usize,
    target: IpwTarget,
    clip: Option<f64>,
}

/// Hajek analytic SE from stacked logistic and weighted-mean estimating equations.
///
/// The IF is the fixed-weight ratio IF plus `B' I⁻¹ x(T-e)`, where `B`
/// differentiates the two weighted means with respect to logistic coefficients.
/// Its derivative depends on the target and is zero beyond clipping thresholds.
/// Trimming membership is locally fixed; all original rows still contribute to
/// the logistic nuisance fit. Bootstrap refits also capture membership changes.
pub(crate) fn hajek_influence_se(
    treatment: &[f64],
    outcome: &[f64],
    weights: &[f64],
    propensity: Option<HajekPropensity<'_>>,
) -> Result<f64, EstimationError> {
    let n = treatment.len();
    if n < 2 || weights.len() != n || outcome.len() != n {
        return Ok(f64::NAN);
    }
    let ncols = propensity.as_ref().map_or(0, |p| p.ncols);
    if let Some(p) = &propensity {
        if p.scores.len() != n || p.design.len() != n * ncols {
            return Err(EstimationError::data_msg("Hajek propensity design/score length mismatch"));
        }
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
        return Ok(f64::NAN);
    }
    let nf = n as f64;
    let mut psi = vec![0.0; n];
    for i in 0..n {
        let (w1, w0) = if treatment[i] > 0.5 { (weights[i], 0.0) } else { (0.0, weights[i]) };
        // Linearized Hajek ratio contributions (E[ψ]=0).
        psi[i] = nf * (w1 / sum_w1) * (outcome[i] - mu1) - nf * (w0 / sum_w0) * (outcome[i] - mu0);
    }

    if let Some(p) = propensity {
        let mut information = vec![0.0; ncols * ncols];
        let mut derivative = vec![0.0; ncols];
        for i in 0..n {
            let e = p.scores[i];
            let clipped = p.clip.is_some_and(|c| e <= c || e >= 1.0 - c);
            let log_weight_derivative = if clipped {
                0.0
            } else {
                match (p.target, treatment[i] > 0.5) {
                    (IpwTarget::Ate | IpwTarget::Custom, true) => -(1.0 - e),
                    (IpwTarget::Ate | IpwTarget::Custom, false) => e,
                    (IpwTarget::Att, true) | (IpwTarget::Atc, false) => 0.0,
                    (IpwTarget::Att, false) => 1.0,
                    (IpwTarget::Atc, true) => -1.0,
                }
            };
            for c in 0..ncols {
                let xc = p.design[c * n + i];
                derivative[c] += psi[i] * log_weight_derivative * xc / nf;
                for d in 0..ncols {
                    information[c * ncols + d] += e * (1.0 - e) * xc * p.design[d * n + i] / nf;
                }
            }
        }
        let Some(alpha) = solve_symmetric_posdef(&mut information, &mut derivative, ncols) else {
            return Err(EstimationError::stats_msg("singular logistic information in Hajek SE"));
        };
        for i in 0..n {
            let adjustment = (0..ncols).map(|c| p.design[c * n + i] * alpha[c]).sum::<f64>();
            psi[i] += adjustment * (treatment[i] - p.scores[i]);
        }
    }

    let mean = psi.iter().sum::<f64>() / nf;
    let var = psi.iter().map(|p| (p - mean).powi(2)).sum::<f64>() / (nf - 1.0);
    // Finite-sample inflation for estimated propensity (p design columns).
    let df = (nf - ncols.max(1) as f64).max(1.0);
    Ok((var / nf * (nf / df)).max(0.0).sqrt())
}

/// Gaussian elimination for a small dense system (propensity score projection).
///
/// Singularity is judged relative to the input matrix's largest absolute diagonal entry
/// (matching `antecedent_stats::gram::invert_square`) so the verdict does not depend on the
/// data's units — an absolute threshold would quietly accept a direction that is singular
/// relative to the Gram matrix's own scale and hand back a garbage adjustment.
pub(crate) fn solve_symmetric_posdef(a: &mut [f64], b: &mut [f64], p: usize) -> Option<Vec<f64>> {
    let mut scale = 0.0_f64;
    for i in 0..p {
        scale = scale.max(a[i * p + i].abs());
    }
    if !(scale.is_finite() && scale > 0.0) {
        return None;
    }
    let tol = 1e-12 * scale;

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
        if best < tol {
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
    use super::solve_symmetric_posdef;

    // Independent case-weight perturbation oracle: refit the logistic MLE and
    // weighted means after infinitesimally changing each observation's mass.
    fn oracle_fit(
        t: &[f64],
        y: &[f64],
        x: &[f64],
        mass: &[f64],
        target: super::IpwTarget,
        clip: Option<f64>,
        trim: Option<f64>,
    ) -> (f64, Vec<f64>) {
        let mut beta = [0.0_f64; 2];
        for _ in 0..100 {
            let mut score = [0.0; 2];
            let mut information = [0.0; 3];
            for i in 0..t.len() {
                let e = 1.0 / (1.0 + (-beta[0] - beta[1] * x[i]).exp());
                let residual = mass[i] * (t[i] - e);
                let v = mass[i] * e * (1.0 - e);
                score[0] += residual;
                score[1] += residual * x[i];
                information[0] += v;
                information[1] += v * x[i];
                information[2] += v * x[i] * x[i];
            }
            let determinant = information[0] * information[2] - information[1].powi(2);
            let step = [
                (information[2] * score[0] - information[1] * score[1]) / determinant,
                (information[0] * score[1] - information[1] * score[0]) / determinant,
            ];
            beta[0] += step[0];
            beta[1] += step[1];
            if step[0].abs().max(step[1].abs()) < 1e-13 {
                break;
            }
        }
        let e: Vec<f64> = x.iter().map(|x| 1.0 / (1.0 + (-beta[0] - beta[1] * x).exp())).collect();
        let mut numerator = [0.0; 2];
        let mut denominator = [0.0; 2];
        for i in 0..t.len() {
            if trim.is_some_and(|c| e[i] < c || e[i] > 1.0 - c) {
                continue;
            }
            let score = clip.map_or(e[i], |c| e[i].clamp(c, 1.0 - c));
            let arm = usize::from(t[i] > 0.5);
            let w = mass[i] * target.weight(t[i], score);
            numerator[arm] += w * y[i];
            denominator[arm] += w;
        }
        (numerator[1] / denominator[1] - numerator[0] / denominator[0], e)
    }

    #[test]
    fn hajek_se_matches_refitted_case_weight_derivatives() {
        let mut t = Vec::new();
        let mut y = Vec::new();
        let mut x = Vec::new();
        for group in 0..5 {
            for row in 0..6 {
                let covariate = f64::from(group) - 2.0;
                let treated = f64::from(row <= group);
                x.push(covariate);
                t.push(treated);
                y.push(covariate + treated * (2.0 + covariate) + 0.2 * f64::from(row * row));
            }
        }
        let n = t.len();
        let mut design = vec![1.0; n];
        design.extend_from_slice(&x);
        for target in [super::IpwTarget::Att, super::IpwTarget::Atc, super::IpwTarget::Ate] {
            for (clip, trim) in [(None, None), (Some(0.25), None), (None, Some(0.2))] {
                let mut mass = vec![1.0; n];
                let (_, e) = oracle_fit(&t, &y, &x, &mass, target, clip, trim);
                let clipped: Vec<f64> =
                    e.iter().map(|&p| clip.map_or(p, |c| p.clamp(c, 1.0 - c))).collect();
                let w = super::compute_ipw_weights(&t, &clipped, &e, target, trim);
                let actual = super::hajek_influence_se(
                    &t,
                    &y,
                    &w,
                    Some(super::HajekPropensity {
                        scores: &e,
                        design: &design,
                        ncols: 2,
                        target,
                        clip,
                    }),
                )
                .unwrap();
                let mut derivatives = Vec::new();
                for row in 0..n {
                    mass[row] = 1.0 + 1e-5;
                    let plus = oracle_fit(&t, &y, &x, &mass, target, clip, trim).0;
                    mass[row] = 1.0 - 1e-5;
                    let minus = oracle_fit(&t, &y, &x, &mass, target, clip, trim).0;
                    mass[row] = 1.0;
                    derivatives.push((plus - minus) / 2e-5);
                }
                let nf = n as f64;
                let expected = (derivatives.iter().map(|d| d * d).sum::<f64>() * nf / (nf - 1.0)
                    * nf
                    / (nf - 2.0))
                    .sqrt();
                assert!(
                    (actual - expected).abs() < 1e-7,
                    "target={target:?} clip={clip:?} trim={trim:?} actual={actual} oracle={expected}"
                );
            }
        }
    }

    #[test]
    fn solve_symmetric_posdef_rejects_badly_scaled_near_singular_matrix() {
        // Rows are nearly parallel at large magnitude — the shape an unnormalized
        // G = S'S / n takes when two propensity-score residual columns are collinear.
        // After one elimination step the remaining pivot is ~1e-6 in absolute terms:
        // comfortably above a fixed 1e-14 absolute threshold (which would wrongly accept
        // this and hand back a garbage adjustment), but far below 1e-12 * scale (~1e-2)
        // once the tolerance is scaled to the matrix's own magnitude (~1e10).
        let mut a = [1e10, 1e10, 1e10, 1e10 + 1e-6];
        let mut b = [1.0, 1.0];
        assert!(solve_symmetric_posdef(&mut a, &mut b, 2).is_none());
    }

    #[test]
    fn solve_symmetric_posdef_still_solves_well_scaled_system() {
        // Sanity check that the relative tolerance doesn't reject ordinary,
        // well-conditioned systems: [4,1;1,3] x = [5,4] ⇒ x = [1,1].
        let mut a = [4.0, 1.0, 1.0, 3.0];
        let mut b = [5.0, 4.0];
        let x = solve_symmetric_posdef(&mut a, &mut b, 2).expect("well-conditioned");
        assert!((x[0] - 1.0).abs() < 1e-12, "x0={}", x[0]);
        assert!((x[1] - 1.0).abs() < 1e-12, "x1={}", x[1]);
    }

    #[test]
    fn solve_symmetric_posdef_rejects_degenerate_scale() {
        // All-zero diagonal (and a non-finite one) leave no usable scale reference.
        let mut a = [0.0, 0.0, 0.0, 0.0];
        let mut b = [1.0, 1.0];
        assert!(solve_symmetric_posdef(&mut a, &mut b, 2).is_none());

        let mut a = [f64::INFINITY, 0.0, 0.0, 1.0];
        let mut b = [1.0, 1.0];
        assert!(solve_symmetric_posdef(&mut a, &mut b, 2).is_none());
    }
}
