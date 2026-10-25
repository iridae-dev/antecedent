//! Confounding sensitivity of a whole target vector, and of a posterior.
//!
//! The scalar analyses of [`crate::evalue_scales`], [`crate::omitted_variable_bias`] and
//! [`crate::rosenbaum`] answer for one coefficient and one confidence interval. A claim about
//! a conditional effect (one coordinate per stratum), a temporal effect (one per horizon), a
//! joint contrast or a mediation vector is explained away as soon as *any* coordinate is, so
//! the assumption range of a vector target is read at its **weakest coordinate**:
//!
//! * the break-even value is the smallest of the coordinates' break-even values, because the
//!   claim "every coordinate keeps its reported sign" first fails there;
//! * each tested value reports the weakest coordinate's adjusted effect (and adjusted limit),
//!   oriented so that a larger value is further from the null.
//!
//! A Bayesian claim reads a posterior, not a confidence interval. [`evalue_posterior_range`]
//! and [`ovb_posterior_range`] carry the posterior through the same bounding factors: each
//! posterior draw is bias-adjusted at every probed confounder strength, and the range reports
//! the weakest coordinate's posterior median together with the **joint posterior probability
//! that every coordinate keeps its reported side of the null** at that strength. The
//! break-even values are the strengths at which the posterior median, and the equal-tailed
//! credible limit nearest the null, reach the null.
//!
//! Each range binds to its [`RefutationTarget`], so a range computed for one target cannot be
//! cited by a claim on another. The confounding structure is an assumption of the analysis,
//! not something the data certify.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::ExecutionContext;
use antecedent_kernels::quantile_type7_sorted as quantile_sorted;
use antecedent_stats::student_t_ppf;

use crate::ValidationError;
use crate::assumption_range::{AssumptionRange, RangePoint, SensitivityAssumption};
use crate::evalue_scales::{
    EValueBounds, EffectInterval, EffectScale, STRENGTH_GRID, bounding_factor, evalue_bounds,
    ratio_to_risk_ratio, smd_to_risk_ratio,
};
use crate::omitted_variable_bias::{LinearAdjustmentSummary, R2DZ_GRID};
use crate::rosenbaum::{SignedRankSensitivity, signed_rank_sensitivity};
use crate::target_native::RefutationTarget;

fn not_applicable(message: &'static str) -> ValidationError {
    ValidationError::NotApplicable { message }
}

fn require_dimension(target: &RefutationTarget, found: usize) -> Result<(), ValidationError> {
    if found == target.dimension() {
        Ok(())
    } else {
        Err(not_applicable(
            "a sensitivity range of a vector target needs one input per coordinate of the target",
        ))
    }
}

/// A risk ratio oriented away from the null: `max(rr, 1 / rr)`.
fn oriented(rr: f64) -> f64 {
    if rr >= 1.0 { rr } else { 1.0 / rr }
}

/// The worst-case adjusted risk ratio at bounding factor `factor`, capped at the null.
fn toward_null(rr: f64, factor: f64) -> f64 {
    if rr >= 1.0 { (rr / factor).max(1.0) } else { (rr * factor).min(1.0) }
}

/// E-values of every coordinate of a vector target as one [`AssumptionRange`] on `target`.
///
/// At each probed confounder strength `s` (risk-ratio scale, equal with treatment and outcome)
/// the tested target is the weakest coordinate's worst-case adjusted risk ratio, oriented away
/// from the null, and the test value is the weakest coordinate's adjusted limit nearest the
/// null (when every coordinate reports an interval). The break-even values are the smallest
/// point E-value and the smallest limit E-value across the coordinates.
///
/// # Errors
///
/// A coordinate count other than the target's dimension, or any error of [`evalue_bounds`].
pub fn evalue_vector_range(
    target: RefutationTarget,
    coordinates: &[EffectInterval],
    scale: EffectScale,
) -> Result<(Vec<EValueBounds>, AssumptionRange), ValidationError> {
    require_dimension(&target, coordinates.len())?;
    let bounds: Vec<EValueBounds> = coordinates
        .iter()
        .map(|interval| evalue_bounds(*interval, scale))
        .collect::<Result<_, _>>()?;
    let all_limits = bounds.iter().all(|b| b.risk_ratio_limit.is_some());
    let tested = STRENGTH_GRID
        .iter()
        .map(|&strength| {
            let factor = bounding_factor(strength);
            let weakest = bounds
                .iter()
                .map(|b| oriented(toward_null(b.risk_ratio, factor)))
                .fold(f64::INFINITY, f64::min);
            let weakest_limit = bounds
                .iter()
                .filter_map(|b| b.risk_ratio_limit)
                .map(|limit| oriented(toward_null(limit, factor)))
                .fold(f64::INFINITY, f64::min);
            RangePoint {
                parameter: strength,
                target: weakest,
                test: (all_limits && weakest_limit.is_finite()).then_some(weakest_limit),
            }
        })
        .collect();
    let break_even = bounds.iter().map(|b| b.point).fold(f64::INFINITY, f64::min);
    let break_even_significance = all_limits
        .then(|| bounds.iter().filter_map(|b| b.limit).fold(f64::INFINITY, f64::min))
        .filter(|value| value.is_finite());
    let range = AssumptionRange::new(
        target,
        SensitivityAssumption::UnmeasuredConfoundingRiskRatio,
        "risk_ratio",
        1.0,
        tested,
        Some(break_even),
        break_even_significance,
    );
    Ok((bounds, range))
}

/// Linear-interpolated quantile of an ascending sample.

fn sorted_copy(values: &[f64]) -> Vec<f64> {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted
}

fn validate_draws(
    target: &RefutationTarget,
    draws: &[Vec<f64>],
    level: f64,
) -> Result<usize, ValidationError> {
    require_dimension(target, draws.len())?;
    let n = draws[0].len();
    if n < 2 || draws.iter().any(|column| column.len() != n) {
        return Err(not_applicable(
            "posterior draws need at least two draws, the same number for every coordinate",
        ));
    }
    if draws.iter().flatten().any(|value| !value.is_finite()) {
        return Err(not_applicable("a posterior draw is not finite"));
    }
    if !(level > 0.0 && level < 1.0) {
        return Err(not_applicable("the credible level must lie in (0, 1)"));
    }
    Ok(n)
}

/// The posterior E-value analysis of a vector target.
#[derive(Clone, Debug, PartialEq)]
pub struct PosteriorEValue {
    /// Posterior median of each coordinate on the risk-ratio scale.
    pub median_risk_ratio: Vec<f64>,
    /// Equal-tailed credible limits of each coordinate on the risk-ratio scale.
    pub credible_interval: Vec<(f64, f64)>,
    /// E-values of each coordinate's posterior median and credible limit nearest the null.
    pub bounds: Vec<EValueBounds>,
    /// `(strength, joint posterior probability)` that every coordinate keeps its reported side
    /// of the null after the worst-case adjustment at that strength.
    pub retention: Vec<(f64, f64)>,
}

/// Convert coordinate-major posterior draws on `scale` to risk-ratio draws.
fn risk_ratio_draws(
    draws: &[Vec<f64>],
    scale: EffectScale,
) -> Result<Vec<Vec<f64>>, ValidationError> {
    draws
        .iter()
        .map(|column| {
            column
                .iter()
                .map(|&value| match scale {
                    EffectScale::StandardizedMeanDifference => Ok(smd_to_risk_ratio(value)),
                    other => ratio_to_risk_ratio(value, other),
                })
                .collect::<Result<Vec<f64>, _>>()
        })
        .collect()
}

/// The E-value analysis of a posterior over a vector target.
///
/// `draws[k][d]` is draw `d` of coordinate `k` on `scale` (a ratio for the ratio scales, the
/// standardized difference for [`EffectScale::StandardizedMeanDifference`]); `level` is the
/// credible level. Each coordinate's reported side of the null is the side of its posterior
/// median. At every probed strength `s` the range reports the weakest coordinate's posterior
/// median adjusted by the bounding factor `B(s)`, and the joint posterior probability that
/// every coordinate's draw is still on its reported side after dividing (multiplying, for a
/// protective effect) by `B(s)`; the probability at `s = 1` is the posterior probability of
/// the reported sign. The break-even values are the smallest E-value of a posterior median and
/// of a credible limit nearest the null.
///
/// # Errors
///
/// Misaligned or non-finite draws, fewer than two draws, a level outside `(0, 1)`, or a
/// non-positive ratio draw.
pub fn evalue_posterior_range(
    target: RefutationTarget,
    draws: &[Vec<f64>],
    scale: EffectScale,
    level: f64,
) -> Result<(PosteriorEValue, AssumptionRange), ValidationError> {
    let n = validate_draws(&target, draws, level)?;
    let ratios = risk_ratio_draws(draws, scale)?;
    let tail = 0.5 * (1.0 - level);
    let mut medians = Vec::with_capacity(ratios.len());
    let mut intervals = Vec::with_capacity(ratios.len());
    let mut bounds = Vec::with_capacity(ratios.len());
    for column in &ratios {
        let sorted = sorted_copy(column);
        let median = quantile_sorted(&sorted, 0.5);
        let (low, high) = (quantile_sorted(&sorted, tail), quantile_sorted(&sorted, 1.0 - tail));
        bounds.push(evalue_bounds(
            EffectInterval::Ratio { estimate: median, lower: Some(low), upper: Some(high) },
            EffectScale::RiskRatio,
        )?);
        medians.push(median);
        intervals.push((low, high));
    }
    // A draw keeps the reported side when it is beyond the null in the direction of the median.
    let keeps = |coordinate: usize, rr: f64, factor: f64| -> bool {
        if medians[coordinate] >= 1.0 { rr / factor > 1.0 } else { rr * factor < 1.0 }
    };
    let mut retention = Vec::with_capacity(STRENGTH_GRID.len());
    let mut tested = Vec::with_capacity(STRENGTH_GRID.len());
    for &strength in &STRENGTH_GRID {
        let factor = bounding_factor(strength);
        let kept =
            (0..n).filter(|&d| (0..ratios.len()).all(|k| keeps(k, ratios[k][d], factor))).count();
        let probability = kept as f64 / n as f64;
        retention.push((strength, probability));
        let weakest = medians
            .iter()
            .map(|&median| oriented(toward_null(median, factor)))
            .fold(f64::INFINITY, f64::min);
        tested.push(RangePoint { parameter: strength, target: weakest, test: Some(probability) });
    }
    let break_even = bounds.iter().map(|b| b.point).fold(f64::INFINITY, f64::min);
    let significance = bounds.iter().filter_map(|b| b.limit).fold(f64::INFINITY, f64::min);
    let range = AssumptionRange::new(
        target,
        SensitivityAssumption::UnmeasuredConfoundingRiskRatio,
        "risk_ratio_posterior",
        1.0,
        tested,
        Some(break_even),
        significance.is_finite().then_some(significance),
    );
    Ok((
        PosteriorEValue {
            median_risk_ratio: medians,
            credible_interval: intervals,
            bounds,
            retention,
        },
        range,
    ))
}

/// The partial-R² analysis of a vector of linear-adjustment coefficients.
///
/// At each probed treatment partial R² (against an outcome partial R² of `r2yz_extreme`) the
/// tested target is the smallest share of the reported estimate left across the coordinates
/// (`adjusted / estimate`, negative once a coordinate is explained past zero), and the test
/// value is the smallest share left of the adjusted `1 - alpha` limit nearest zero
/// (`direction * limit / |estimate|`, positive while that limit excludes zero). The break-even
/// values are the smallest robustness values `RV_q` and `RV_{q,alpha}` across the coordinates.
///
/// # Errors
///
/// A coordinate count other than the target's dimension, `q` not positive, `alpha` outside
/// `(0, 1)` or `r2yz_extreme` outside `(0, 1]`.
pub fn ovb_vector_range(
    target: RefutationTarget,
    summaries: &[LinearAdjustmentSummary],
    q: f64,
    alpha: f64,
    r2yz_extreme: f64,
) -> Result<AssumptionRange, ValidationError> {
    require_dimension(&target, summaries.len())?;
    if !(q > 0.0
        && q.is_finite()
        && alpha > 0.0
        && alpha < 1.0
        && r2yz_extreme > 0.0
        && r2yz_extreme <= 1.0)
    {
        return Err(not_applicable(
            "q must be positive, alpha must lie in (0, 1) and the extreme outcome partial R2 in \
             (0, 1]",
        ));
    }
    let tested = R2DZ_GRID
        .iter()
        .map(|&r2dz| {
            let mut remaining = f64::INFINITY;
            let mut remaining_limit = f64::INFINITY;
            for summary in summaries {
                let z = student_t_ppf(1.0 - alpha / 2.0, summary.dof - 1.0);
                let estimate = summary.adjusted_estimate(r2dz, r2yz_extreme);
                let se = summary.adjusted_standard_error(r2dz, r2yz_extreme);
                let direction = if summary.estimate >= 0.0 { 1.0 } else { -1.0 };
                let limit = estimate - direction * z * se;
                let scale = summary.estimate.abs().max(f64::MIN_POSITIVE);
                remaining = remaining.min(direction * estimate / scale);
                remaining_limit = remaining_limit.min(direction * limit / scale);
            }
            RangePoint { parameter: r2dz, target: remaining, test: Some(remaining_limit) }
        })
        .collect();
    let break_even =
        summaries.iter().map(|s| s.robustness_value(q, None)).fold(f64::INFINITY, f64::min);
    let significance =
        summaries.iter().map(|s| s.robustness_value(q, Some(alpha))).fold(f64::INFINITY, f64::min);
    Ok(AssumptionRange::new(
        target,
        SensitivityAssumption::UnmeasuredConfoundingPartialR2,
        "partial_r2",
        0.0,
        tested,
        Some(break_even),
        Some(significance),
    ))
}

/// Posterior draws of a vector of linear-adjustment coefficients with the scale of their
/// omitted-variable bias.
#[derive(Clone, Debug)]
pub struct OvbPosterior<'a> {
    /// `coefficients[k][d]`: draw `d` of coordinate `k`'s treatment coefficient.
    pub coefficients: &'a [Vec<f64>],
    /// `bias_scale[k][d]`: `sd(Y_k | D, X) / sd(D | X)` in draw `d`, the factor that turns the
    /// partial R²s into a coefficient bias, `bias = scale * sqrt(r2yz * r2dz / (1 - r2dz))`
    /// (Cinelli and Hazlett 2020). It comes from the residual-scale posterior of the outcome
    /// model and the residual scale of the treatment model.
    pub bias_scale: &'a [Vec<f64>],
}

/// The partial-R² analysis of a posterior over a vector of linear-adjustment coefficients.
///
/// The worst-case bias of a confounder with partial R²s `(r2dz, r2yz)` moves each draw
/// toward the null by `bias_scale * sqrt(r2yz * r2dz / (1 - r2dz))`. At every probed treatment
/// partial R² (against `r2yz_extreme`) the range reports the smallest share of the posterior
/// median left across the coordinates and the joint posterior probability that every
/// coordinate keeps its reported side. The break-even values are the smallest equal partial R²
/// (`r2dz = r2yz`) at which a coordinate's adjusted posterior median, and its adjusted
/// credible limit nearest the null, reach zero.
///
/// # Errors
///
/// Misaligned or non-finite draws, a non-positive bias scale, a level outside `(0, 1)`, or
/// `r2yz_extreme` outside `(0, 1]`.
pub fn ovb_posterior_range(
    target: RefutationTarget,
    posterior: &OvbPosterior<'_>,
    level: f64,
    r2yz_extreme: f64,
) -> Result<AssumptionRange, ValidationError> {
    let n = validate_draws(&target, posterior.coefficients, level)?;
    if posterior.bias_scale.len() != posterior.coefficients.len()
        || posterior.bias_scale.iter().any(|column| column.len() != n)
        || posterior.bias_scale.iter().flatten().any(|v| !(v.is_finite() && *v >= 0.0))
    {
        return Err(not_applicable(
            "the bias scale needs one non-negative finite draw per coefficient draw",
        ));
    }
    if !(r2yz_extreme > 0.0 && r2yz_extreme <= 1.0) {
        return Err(not_applicable("the extreme outcome partial R2 must lie in (0, 1]"));
    }
    let tail = 0.5 * (1.0 - level);
    let k = posterior.coefficients.len();
    let direction: Vec<f64> = posterior
        .coefficients
        .iter()
        .map(|column| if quantile_sorted(&sorted_copy(column), 0.5) >= 0.0 { 1.0 } else { -1.0 })
        .collect();
    let medians: Vec<f64> = (0..k)
        .map(|j| quantile_sorted(&sorted_copy(&posterior.coefficients[j]), 0.5).abs())
        .collect();
    if medians.iter().any(|m| *m <= 0.0) {
        return Err(not_applicable(
            "a coordinate's posterior median is zero: nothing to explain away",
        ));
    }
    // Oriented adjusted draws of coordinate `j` at bias multiplier `shift`.
    let adjusted = |j: usize, shift: f64| -> Vec<f64> {
        (0..n)
            .map(|d| {
                direction[j] * posterior.coefficients[j][d] - posterior.bias_scale[j][d] * shift
            })
            .collect()
    };
    let shift = |r2dz: f64, r2yz: f64| (r2yz * r2dz / (1.0 - r2dz)).sqrt();
    let tested = R2DZ_GRID
        .iter()
        .map(|&r2dz| {
            let multiplier = shift(r2dz, r2yz_extreme);
            let mut remaining = f64::INFINITY;
            let mut kept = vec![true; n];
            for j in 0..k {
                let draws = adjusted(j, multiplier);
                remaining = remaining.min(quantile_sorted(&sorted_copy(&draws), 0.5) / medians[j]);
                for (flag, value) in kept.iter_mut().zip(&draws) {
                    *flag &= *value > 0.0;
                }
            }
            let probability = kept.iter().filter(|flag| **flag).count() as f64 / n as f64;
            RangePoint { parameter: r2dz, target: remaining, test: Some(probability) }
        })
        .collect();
    // Smallest equal partial R2 `r` at which the `p`-quantile of the oriented adjusted draws
    // reaches zero; the quantile is non-increasing in `r`.
    let break_even_at = |j: usize, p: f64| -> Option<f64> {
        let quantile = |r: f64| quantile_sorted(&sorted_copy(&adjusted(j, shift(r, r))), p);
        if quantile(0.0) <= 0.0 {
            return Some(0.0);
        }
        let upper = 1.0 - 1e-12;
        if quantile(upper) > 0.0 {
            return None;
        }
        let (mut lo, mut hi) = (0.0_f64, upper);
        for _ in 0..80 {
            let mid = 0.5 * (lo + hi);
            if quantile(mid) > 0.0 {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        Some(hi)
    };
    let smallest = |p: f64| -> Option<f64> {
        (0..k)
            .map(|j| break_even_at(j, p))
            .collect::<Option<Vec<f64>>>()
            .map(|values| values.into_iter().fold(f64::INFINITY, f64::min))
    };
    // The limit nearest the null of an oriented coordinate is its lower credible limit.
    Ok(AssumptionRange::new(
        target,
        SensitivityAssumption::UnmeasuredConfoundingPartialR2,
        "partial_r2_posterior",
        0.0,
        tested,
        smallest(0.5),
        smallest(tail),
    ))
}

/// The signed-rank sensitivity of a vector of matched designs as one [`AssumptionRange`].
///
/// `differences[k]` are the matched-pair differences behind coordinate `k`. At every probed
/// `Γ` the tested target (and test) is the largest upper-bound p-value across the coordinates,
/// the weakest evidence; the break-even value is the smallest `Γ*` across the coordinates, and
/// is absent when some coordinate does not reject even at `Γ = 1`.
///
/// # Errors
///
/// A coordinate count other than the target's dimension, or any error of
/// [`signed_rank_sensitivity`].
pub fn rosenbaum_vector_range(
    target: RefutationTarget,
    differences: &[Vec<f64>],
    gammas: &[f64],
    alpha: f64,
    ctx: Option<&ExecutionContext>,
) -> Result<(Vec<SignedRankSensitivity>, AssumptionRange), ValidationError> {
    require_dimension(&target, differences.len())?;
    let analyses: Vec<SignedRankSensitivity> = differences
        .iter()
        .map(|d| signed_rank_sensitivity(d, gammas, alpha, ctx))
        .collect::<Result<_, _>>()?;
    let tested = (0..gammas.len())
        .map(|g| {
            let worst = analyses.iter().map(|a| a.grid[g].1).fold(0.0_f64, f64::max);
            RangePoint { parameter: gammas[g], target: worst, test: Some(worst) }
        })
        .collect();
    let break_even = analyses
        .iter()
        .map(|a| a.gamma_break_even)
        .collect::<Option<Vec<f64>>>()
        .map(|values| values.into_iter().fold(f64::INFINITY, f64::min));
    let range = AssumptionRange::new(
        target,
        SensitivityAssumption::HiddenBiasInMatchedAssignment,
        "gamma",
        1.0,
        tested,
        None,
        break_even,
    );
    Ok((analyses, range))
}

#[cfg(test)]
#[allow(
    clippy::float_cmp,
    reason = "tests locate exact values copied verbatim from the declared sensitivity grids"
)]
mod tests {
    use super::*;
    use crate::target_native::{TargetFamily, TargetQuadrant};
    use crate::{evalue_of_risk_ratio, evalue_range};

    fn target(family: TargetFamily, quadrant: TargetQuadrant, labels: &[&str]) -> RefutationTarget {
        RefutationTarget::new(family, quadrant, labels.iter().copied(), [3_u8; 32]).unwrap()
    }

    fn ratio(estimate: f64, lower: f64, upper: f64) -> EffectInterval {
        EffectInterval::Ratio { estimate, lower: Some(lower), upper: Some(upper) }
    }

    #[test]
    fn the_vector_evalue_range_is_read_at_the_weakest_coordinate() {
        let stratum = target(
            TargetFamily::ConditionalEffect,
            TargetQuadrant::STATIC_FREQUENTIST,
            &["young", "old"],
        );
        let strong = ratio(3.9, 1.8, 8.7);
        let weak = ratio(2.0, 1.2, 3.3);
        let (bounds, range) =
            evalue_vector_range(stratum.clone(), &[strong, weak], EffectScale::RiskRatio).unwrap();
        assert!((bounds[0].point - evalue_of_risk_ratio(3.9)).abs() < 1e-12);
        assert!((bounds[1].point - evalue_of_risk_ratio(2.0)).abs() < 1e-12);
        // The claim "both strata keep their sign" breaks at the weaker stratum's E-value.
        assert!((range.break_even.unwrap() - evalue_of_risk_ratio(2.0)).abs() < 1e-12);
        assert!((range.break_even_significance.unwrap() - evalue_of_risk_ratio(1.2)).abs() < 1e-12);
        // The weakest adjusted risk ratio at strength 2 is the weak stratum's 2 / B(2).
        let at_two = range.tested.iter().find(|p| p.parameter == 2.0).unwrap();
        assert!((at_two.target - 2.0 / bounding_factor(2.0)).abs() < 1e-12);
        assert!((at_two.test.unwrap() - (1.2 / bounding_factor(2.0)).max(1.0)).abs() < 1e-12);
        // One coordinate reproduces the scalar analysis exactly.
        let single =
            target(TargetFamily::ScalarAverageEffect, TargetQuadrant::STATIC_FREQUENTIST, &["ate"]);
        let (_, vector) =
            evalue_vector_range(single.clone(), &[strong], EffectScale::RiskRatio).unwrap();
        let (_, scalar) = evalue_range(single, strong, EffectScale::RiskRatio).unwrap();
        assert_eq!(vector.tested, scalar.tested);
        assert_eq!(vector.break_even, scalar.break_even);
        // The range is about its target, and the wrong number of coordinates is refused.
        assert!(vector.bound_to(&stratum).is_err());
        assert!(
            evalue_vector_range(stratum, &[strong], EffectScale::RiskRatio).is_err(),
            "one interval for two strata"
        );
    }

    /// Posterior draws of a log risk ratio spread uniformly around `center` with half-width
    /// `half`, as a risk-ratio sample.
    fn spread(center: f64, half: f64, n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| (center.ln() + half * (2.0 * i as f64 / (n - 1) as f64 - 1.0)).exp())
            .collect()
    }

    #[test]
    fn the_posterior_evalue_range_carries_the_draws_through_the_bounding_factor() {
        let horizons =
            target(TargetFamily::TemporalEffect, TargetQuadrant::TEMPORAL_BAYESIAN, &["h1", "h2"]);
        let draws = vec![spread(3.0, 0.15, 101), spread(1.6, 0.25, 101)];
        let (summary, range) =
            evalue_posterior_range(horizons.clone(), &draws, EffectScale::RiskRatio, 0.95).unwrap();
        // Medians of an odd, symmetric-in-log sample are the center.
        assert!((summary.median_risk_ratio[0] - 3.0).abs() < 1e-12);
        assert!((summary.median_risk_ratio[1] - 1.6).abs() < 1e-12);
        // The credible limit nearest the null of the second horizon is its 2.5% quantile.
        // The 2.5% quantile of 101 draws interpolates halfway between the third and fourth.
        let lower = 0.5 * (draws[1][2] + draws[1][3]);
        assert!((summary.credible_interval[1].0 - lower).abs() < 1e-12);
        assert!((range.break_even.unwrap() - evalue_of_risk_ratio(1.6)).abs() < 1e-9);
        assert!(
            (range.break_even_significance.unwrap() - evalue_of_risk_ratio(lower)).abs() < 1e-9
        );
        // Retention against an independent count of the draws.
        for &(strength, probability) in &summary.retention {
            let factor = strength * strength / (2.0 * strength - 1.0);
            let kept = (0..101).filter(|&d| draws.iter().all(|c| c[d] / factor > 1.0)).count();
            assert!((probability - kept as f64 / 101.0).abs() < 1e-12, "strength {strength}");
        }
        // No confounding keeps every draw; a strong one removes the second horizon's.
        assert!((summary.retention[0].1 - 1.0).abs() < 1e-12);
        assert!(summary.retention.last().unwrap().1 < 1e-12);
        // Retention is non-increasing in the confounder strength.
        assert!(summary.retention.windows(2).all(|w| w[1].1 <= w[0].1 + 1e-12));
        assert_eq!(range.parameter.as_ref(), "risk_ratio_posterior");
        assert!(range.bound_to(&horizons).is_ok());
        // A protective coordinate is read on its own side.
        let protective =
            target(TargetFamily::ConditionalEffect, TargetQuadrant::STATIC_BAYESIAN, &["only"]);
        let (flipped, _) = evalue_posterior_range(
            protective,
            &[spread(0.5, 0.1, 101)],
            EffectScale::RiskRatio,
            0.9,
        )
        .unwrap();
        assert!((flipped.bounds[0].point - evalue_of_risk_ratio(0.5)).abs() < 1e-9);
        assert!((flipped.retention[0].1 - 1.0).abs() < 1e-12);
    }

    #[test]
    fn the_standardized_scale_converts_draws_and_bad_draws_are_refused() {
        let one = target(TargetFamily::ConditionalEffect, TargetQuadrant::STATIC_BAYESIAN, &["d"]);
        let draws: Vec<f64> = (0..101).map(|i| 0.4 + 0.1 * (i as f64 / 100.0 - 0.5)).collect();
        let (summary, _) = evalue_posterior_range(
            one.clone(),
            &[draws],
            EffectScale::StandardizedMeanDifference,
            0.9,
        )
        .unwrap();
        assert!((summary.median_risk_ratio[0] - smd_to_risk_ratio(0.4)).abs() < 1e-12);
        assert!(
            evalue_posterior_range(one.clone(), &[vec![1.0]], EffectScale::RiskRatio, 0.9).is_err()
        );
        assert!(
            evalue_posterior_range(one.clone(), &[vec![1.0, -2.0]], EffectScale::RiskRatio, 0.9)
                .is_err(),
            "a ratio draw must be positive"
        );
        assert!(
            evalue_posterior_range(one, &[vec![1.0, 2.0]], EffectScale::RiskRatio, 1.5).is_err()
        );
    }

    #[test]
    fn the_vector_ovb_range_reports_the_smallest_robustness_value() {
        let strata = target(
            TargetFamily::ConditionalEffect,
            TargetQuadrant::STATIC_FREQUENTIST,
            &["a", "b"],
        );
        let strong = LinearAdjustmentSummary::new(0.0973, 0.0233, 783.0).unwrap();
        let weak = LinearAdjustmentSummary::new(-0.04, 0.02, 500.0).unwrap();
        let range = ovb_vector_range(strata.clone(), &[strong, weak], 1.0, 0.05, 1.0).unwrap();
        let expected = strong.robustness_value(1.0, None).min(weak.robustness_value(1.0, None));
        assert!((range.break_even.unwrap() - expected).abs() < 1e-12);
        assert!(range.break_even.unwrap() <= strong.robustness_value(1.0, None));
        let first = range.tested.first().unwrap();
        assert!((first.target - 1.0).abs() < 1e-12, "no confounding leaves the whole estimate");
        // Past the weak coordinate's break-even partial R2 the claim is explained away.
        let past = range.tested.iter().find(|p| p.parameter >= 0.5).unwrap();
        assert!(past.target < 1.0);
        let relative = |summary: LinearAdjustmentSummary, r2dz: f64| {
            let direction = summary.estimate.signum();
            direction * summary.adjusted_estimate(r2dz, 1.0) / summary.estimate.abs()
        };
        for point in range.tested.iter() {
            let expected = relative(strong, point.parameter).min(relative(weak, point.parameter));
            assert!((point.target - expected).abs() < 1e-12, "r2dz {}", point.parameter);
        }
        assert!(ovb_vector_range(strata, &[strong], 1.0, 0.05, 1.0).is_err());
    }

    /// Deterministic standard-normal-like draws: a fixed shape, rescaled.
    fn draws_around(center: f64, sd: f64, n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| {
                let u = (i as f64 + 0.5) / n as f64;
                center + sd * antecedent_kernels::norm_inv(u)
            })
            .collect()
    }

    #[test]
    fn the_posterior_ovb_range_moves_each_draw_by_its_own_bias_scale() {
        let horizons =
            target(TargetFamily::TemporalEffect, TargetQuadrant::TEMPORAL_BAYESIAN, &["h1", "h2"]);
        let n = 400;
        let coefficients = vec![draws_around(0.5, 0.05, n), draws_around(-0.3, 0.05, n)];
        let scale = vec![vec![1.2; n], vec![0.8; n]];
        let posterior = OvbPosterior { coefficients: &coefficients, bias_scale: &scale };
        let range = ovb_posterior_range(horizons.clone(), &posterior, 0.9, 1.0).unwrap();
        assert!(range.bound_to(&horizons).is_ok());
        // No confounding: the median is untouched and the draws keep their side.
        let first = range.tested.first().unwrap();
        assert!((first.target - 1.0).abs() < 1e-9);
        assert!(first.test.unwrap() > 0.999);
        // Independent re-derivation of the break-even at the median: coordinate j reaches zero
        // when scale * r / sqrt(1 - r) = |median| (equal partial R2s, r2yz = r2dz = r).
        let reaches = |median: f64, scale: f64| {
            // Solve r^2 / (1 - r) = (median / scale)^2 for r.
            let c = (median / scale).powi(2);
            0.5 * (-c + (c * c + 4.0 * c).sqrt())
        };
        let expected = reaches(0.5, 1.2).min(reaches(0.3, 0.8));
        assert!(
            (range.break_even.unwrap() - expected).abs() < 5e-3,
            "{} vs {expected}",
            range.break_even.unwrap()
        );
        // The credible limit is reached no later than the median.
        assert!(range.break_even_significance.unwrap() <= range.break_even.unwrap() + 1e-9);
        assert_eq!(range.parameter.as_ref(), "partial_r2_posterior");
        // Misaligned bias scales are refused.
        let short = vec![vec![1.0; n], vec![1.0; n - 1]];
        let bad = OvbPosterior { coefficients: &coefficients, bias_scale: &short };
        assert!(ovb_posterior_range(horizons, &bad, 0.9, 1.0).is_err());
    }

    #[test]
    fn the_vector_rosenbaum_range_is_the_weakest_coordinate() {
        let strata = target(
            TargetFamily::ConditionalEffect,
            TargetQuadrant::STATIC_FREQUENTIST,
            &["a", "b"],
        );
        let strong: Vec<f64> = (1..=14).map(|i| 1.0 + 0.2 * f64::from(i)).collect();
        let weaker: Vec<f64> = (1..=14)
            .map(|i| if i % 3 == 0 { -0.3 * f64::from(i) } else { 0.5 * f64::from(i) })
            .collect();
        let gammas = [1.0, 1.5, 2.0, 3.0];
        let (analyses, range) =
            rosenbaum_vector_range(strata, &[strong.clone(), weaker.clone()], &gammas, 0.05, None)
                .unwrap();
        for (g, point) in range.tested.iter().enumerate() {
            let expected = analyses[0].grid[g].1.max(analyses[1].grid[g].1);
            assert!((point.target - expected).abs() < 1e-15);
        }
        let expected = match (analyses[0].gamma_break_even, analyses[1].gamma_break_even) {
            (Some(a), Some(b)) => Some(a.min(b)),
            _ => None,
        };
        assert_eq!(range.break_even_significance, expected);
        // A coordinate that does not reject at Gamma = 1 leaves the vector claim without a
        // break-even value.
        let null: Vec<f64> = (1..=14).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
        let strata = target(
            TargetFamily::ConditionalEffect,
            TargetQuadrant::STATIC_FREQUENTIST,
            &["a", "b"],
        );
        let (_, flat) =
            rosenbaum_vector_range(strata, &[strong, null], &gammas, 0.05, None).unwrap();
        assert_eq!(flat.break_even_significance, None);
    }
}
