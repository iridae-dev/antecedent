//! Extracted stage of the parent estimator module.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

/// Compute AIPW per-unit IF values for ATE / ATT / ATC.
#[cfg(test)]
pub(super) fn aipw_psi(
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
pub(super) fn aipw_psi_weighted(
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
                .filter(|(_, t)| TreatmentArm::is_treated_indicator(**t))
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
pub(super) fn center_population_psi(
    psi: &mut [f64],
    treatment: &[f64],
    target: &TargetPopulation,
    estimate: f64,
    weights: Option<&[f64]>,
) {
    let n = treatment.len() as f64;
    let weight = |i: usize| weights.map_or(1.0, |w| w[i]);
    let arm_indicator = |t: f64, treated: bool| -> f64 {
        if TreatmentArm::is_treated_indicator(t) == treated { 1.0 } else { 0.0 }
    };
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
pub(super) fn correct_aipw_psi_for_nuisances(
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
pub(super) struct AipwNuisanceFit<'a> {
    pub(super) treatment: &'a [f64],
    pub(super) outcome: &'a [f64],
    /// Clipped propensity used in ψ.
    pub(super) e_used: &'a [f64],
    /// Raw fitted propensity (the logistic MLE's own scores).
    pub(super) e_raw: &'a [f64],
    pub(super) mu0: &'a [f64],
    pub(super) mu1: &'a [f64],
    pub(super) design: &'a [f64],
    pub(super) ncols: usize,
    /// Sampling weights of the rows (mean one), when declared.
    pub(super) weights: Option<&'a [f64]>,
}

/// Overlap report of the treated arm's cross-fitted propensities, when the table carries one.
pub(super) fn treated_arm_overlap_report(
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

/// The rows a full-sample AIPW contrast uses: the common-support rows a trim rule retains (all
/// rows when there is none), with the sampling weights restricted to them and renormalized to
/// mean one, so a trimmed fit estimates the effect in the retained weighted population.
pub(super) struct ContrastRows<'a> {
    /// Indices of the retained rows, or `None` when every row is used.
    pub(super) retained: Option<Vec<usize>>,
    /// Sampling weights of the used rows.
    pub(super) sampling: Option<RetainedSamplingWeights>,
    /// Column-major design of the used rows.
    pub(super) design: Cow<'a, [f64]>,
    /// Treatment of the used rows.
    pub(super) treatment: Cow<'a, [f64]>,
    /// Outcome of the used rows.
    pub(super) outcome: Cow<'a, [f64]>,
    /// Clipped propensity of the used rows.
    pub(super) propensity: Cow<'a, [f64]>,
}

impl<'a> ContrastRows<'a> {
    /// Select the rows of `problem` under the overlap policy's trim rule, on the raw scores of
    /// `model` so the excluded units match the overlap report's common-support claim.
    pub(super) fn select(
        problem: &'a PreparedPropensityProblem,
        model: &'a PropensityModel,
    ) -> Result<Self, EstimationError> {
        let retained = trim_retained_rows(&model.fit.scores, trim_of(problem.overlap))?;
        let ncols = problem.design_ncols;
        let sampling = match (&retained, problem.sampling_weights.as_ref()) {
            (Some(idx), Some(sampling)) => Some(sampling.subset(idx)),
            (None, Some(sampling)) => Some(sampling.clone()),
            (_, None) => None,
        };
        let (design, treatment, outcome, propensity) = match &retained {
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
        Ok(Self { retained, sampling, design, treatment, outcome, propensity })
    }

    /// Sampling weights of the used rows, when declared.
    pub(super) fn weights(&self) -> Option<&[f64]> {
        self.sampling.as_ref().map(|s| &*s.weights)
    }
}
