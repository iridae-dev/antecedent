use super::{
    CandidateDesign, DesignError, EffectWidthContext, ModelLoglikDraws, observation_reliability,
};
use antecedent_core::{CausalRng, ModelId};
use antecedent_kernels::{chol_solve_into, cholesky_spd_into};

/// Standard error of the treatment coefficient, `sqrt(σ² · [(XᵀX)⁻¹]_tt)`.
///
/// `what` names the Gram in error messages. A Gram that cannot be inverted is an error:
/// reporting it as "no information gain" would rank every candidate as an exact tie.
fn treatment_se(
    xtx: &[f64],
    p: usize,
    treatment_col: usize,
    sigma2: f64,
    what: &str,
) -> Result<f64, DesignError> {
    if p == 0 || xtx.len() != p * p {
        return Err(DesignError::Shape(format!(
            "{what}: Gram must be a non-empty square p×p matrix (got {} entries)",
            xtx.len()
        )));
    }
    if treatment_col >= p {
        return Err(DesignError::Shape(format!(
            "{what}: treatment_col {treatment_col} out of range for p = {p}"
        )));
    }
    if !(sigma2.is_finite() && sigma2 > 0.0) {
        return Err(DesignError::Config(format!(
            "{what}: residual variance must be positive and finite (got {sigma2})"
        )));
    }
    // `[(XᵀX)⁻¹]_tt` is the t-th component of the solution of `XᵀX z = e_t`: one Cholesky
    // factorization and one pair of triangular solves, not the full inverse.
    let mut factor = vec![0.0; p * p];
    cholesky_spd_into(xtx, p, &mut factor).map_err(|error| {
        DesignError::Numerical(format!(
            "{what}: Gram is singular or not positive definite ({error:?})"
        ))
    })?;
    let mut unit = vec![0.0; p];
    unit[treatment_col] = 1.0;
    let (mut forward, mut solution) = (vec![0.0; p], vec![0.0; p]);
    chol_solve_into(&factor, p, &unit, &mut forward, &mut solution);
    Ok((sigma2 * solution[treatment_col].max(0.0)).sqrt())
}

fn gram_side_len(xtx: &[f64]) -> usize {
    let n2 = xtx.len();
    let mut k = 0usize;
    while k * k < n2 {
        k += 1;
    }
    if k * k == n2 { k } else { 0 }
}

/// Baseline `(p, se₀)` of the treatment coefficient.
pub(super) fn baseline_treatment_se(ew: &EffectWidthContext) -> Result<(usize, f64), DesignError> {
    let p = gram_side_len(&ew.xtx);
    let se0 = treatment_se(&ew.xtx, p, ew.treatment_col, ew.sigma2, "baseline")?;
    Ok((p, se0))
}

/// Expand `p×p` Gram by appending one column with given cross-products.
fn expand_gram(xtx: &[f64], p: usize, cross: &[f64], self_dot: f64) -> Option<Vec<f64>> {
    if cross.len() != p || xtx.len() != p * p {
        return None;
    }
    let p1 = p + 1;
    let mut out = vec![0.0; p1 * p1];
    for i in 0..p {
        for j in 0..p {
            out[i * p1 + j] = xtx[i * p + j];
        }
        out[i * p1 + p] = cross[i];
        out[p * p1 + i] = cross[i];
    }
    out[p * p1 + p] = self_dot;
    Some(out)
}

/// Signed reduction `se₀ − se₁` of the treatment-effect standard error.
///
/// A design that *raises* the SE (a collinear covariate, a noisier environment) scores
/// negative and ranks below a neutral one. `0` means the candidate has no modelled effect
/// (no sample increment, no matching design-analysis columns, no environment / intervention
/// Gram supplied); a malformed or singular Gram is an error, not a zero.
pub(super) fn effect_width_reduction(
    candidate: &CandidateDesign,
    ew: &EffectWidthContext,
) -> Result<f64, DesignError> {
    effect_width_reduction_from(candidate, ew, baseline_treatment_se(ew)?)
}

/// [`effect_width_reduction`] against a baseline `(p, se₀)` the caller already solved.
pub(super) fn effect_width_reduction_from(
    candidate: &CandidateDesign,
    ew: &EffectWidthContext,
    (p, se0): (usize, f64),
) -> Result<f64, DesignError> {
    match candidate {
        CandidateDesign::IncreaseSamplingRate(s) => {
            if s.additional_samples == 0 {
                return Ok(0.0);
            }
            // XtX scales with n; SE scales as 1/sqrt(n).
            let n1 = (ew.n + s.additional_samples) as f64;
            let n0 = ew.n.max(1) as f64;
            let se1 = se0 * (n0 / n1).sqrt();
            Ok(se0 - se1)
        }
        CandidateDesign::ObserveEnvironment(e) => {
            if let Some(grams) = ew.environment_grams.as_ref() {
                if let Some(spec) = grams.iter().find(|s| s.environment == e.environment) {
                    let p_env = gram_side_len(&spec.xtx);
                    if p_env != p || spec.xtx.len() != ew.xtx.len() {
                        return Err(DesignError::Shape(format!(
                            "environment {:?} Gram has {} entries, expected {}",
                            e.environment,
                            spec.xtx.len(),
                            ew.xtx.len()
                        )));
                    }
                    let mut pooled = ew.xtx.to_vec();
                    for (a, b) in pooled.iter_mut().zip(spec.xtx.iter()) {
                        *a += *b;
                    }
                    let sigma2 = spec.sigma2.unwrap_or(ew.sigma2);
                    let se1 =
                        treatment_se(&pooled, p, ew.treatment_col, sigma2, "pooled environment")?;
                    return Ok(se0 - se1);
                }
            }
            // No env Gram: fall back to additional-row isotropic scaling for SE only.
            if e.additional_rows == 0 {
                return Ok(0.0);
            }
            let n1 = (ew.n + e.additional_rows) as f64;
            let n0 = ew.n.max(1) as f64;
            let se1 = se0 * (n0 / n1).sqrt();
            Ok(se0 - se1)
        }
        CandidateDesign::Measure(plan) => {
            let Some(specs) = ew.measure_columns.as_ref() else {
                // No design-analysis columns → no modelled SE change.
                return Ok(0.0);
            };
            let mut xtx = ew.xtx.to_vec();
            let mut cur_p = p;
            let mut sigma2 = ew.sigma2;
            let mut matched = 0usize;
            for v in plan.variables.iter() {
                let Some(spec) = specs.iter().find(|s| s.variable == *v) else {
                    continue;
                };
                let Some(expanded) = expand_gram(&xtx, cur_p, &spec.cross, spec.self_dot) else {
                    return Err(DesignError::Shape(format!(
                        "measure column for variable {v:?} has {} cross-products, expected {cur_p}",
                        spec.cross.len()
                    )));
                };
                xtx = expanded;
                cur_p += 1;
                if let Some(s2) = spec.sigma2_after {
                    sigma2 = s2;
                }
                matched += 1;
            }
            if matched == 0 {
                return Ok(0.0);
            }
            let se1 = treatment_se(&xtx, cur_p, ew.treatment_col, sigma2, "post-measurement")?;
            Ok(se0 - se1)
        }
        CandidateDesign::Intervene(_) => {
            let Some(design) = ew.intervention_design.as_ref() else {
                return Ok(0.0);
            };
            let p1 = gram_side_len(&design.xtx);
            if p1 != p {
                return Err(DesignError::Shape(format!(
                    "intervention design Gram has {} entries, expected {}",
                    design.xtx.len(),
                    ew.xtx.len()
                )));
            }
            let se1 = treatment_se(
                &design.xtx,
                p1,
                ew.treatment_col,
                design.sigma2,
                "post-intervention",
            )?;
            Ok(se0 - se1)
        }
    }
}

/// Table rows `(i, j)` of every pair drawn from `models` that is present in `ll`.
pub(super) fn model_pair_rows(ll: &ModelLoglikDraws, models: &[ModelId]) -> Vec<(usize, usize)> {
    let row_of: Vec<Option<usize>> =
        models.iter().map(|m| ll.models.iter().position(|x| x == m)).collect();
    let mut pairs = Vec::new();
    for i in 0..models.len() {
        for j in (i + 1)..models.len() {
            if let (Some(ri), Some(rj)) = (row_of[i], row_of[j]) {
                pairs.push((ri, rj));
            }
        }
    }
    pairs
}

pub(super) fn model_distinguish_score(
    candidate: &CandidateDesign,
    ll: &ModelLoglikDraws,
    n_models: usize,
    pairs: &[(usize, usize)],
    rng: &mut CausalRng,
) -> f64 {
    if n_models < 2 || ll.n_draws == 0 {
        return 0.0;
    }
    #[allow(
        clippy::cast_possible_truncation,
        reason = "the draw is reduced modulo `n_draws`, a usize, so the result is a valid index; only uniform high bits are dropped on 32-bit targets"
    )]
    let draw = (rng.next_u64() as usize) % ll.n_draws;
    let strength = observation_reliability(candidate);
    // Expected absolute log-score gap over the requested model pairs, scaled by evidence strength.
    let mut gap = 0.0;
    for &(ri, rj) in pairs {
        let a = ll.loglik[ri * ll.n_draws + draw];
        let b = ll.loglik[rj * ll.n_draws + draw];
        gap += (a - b).abs();
    }
    if pairs.is_empty() { 0.0 } else { strength * gap / pairs.len() as f64 }
}
