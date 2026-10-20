//! Model-assisted population intervals for linear-outcome AIPW scores.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::propensity::PreparedPropensityProblem;
use crate::{EffectEstimate, EstimationError};

/// A population ATE interval under independent homoskedastic Gaussian arm-linear outcomes.
///
/// The outcome coefficient includes every training-fold fit. The optional finite ridge
/// mean correction projects that coefficient onto the arm-linear target, preserving its
/// orthogonal residual component. The conditional noise variance is exact under the
/// declared model; population heterogeneity and mass use an asymptotic Gaussian score.
#[derive(Clone, Debug)]
pub struct PopulationAipwInterval {
    /// The model-assisted point and its combined noise-plus-target standard error.
    pub estimate: EffectEstimate,
    /// Lower confidence endpoint.
    pub lower: f64,
    /// Upper confidence endpoint.
    pub upper: f64,
    /// Confidence level.
    pub level: f64,
    /// Change from the original score point, correcting finite ridge mean shrinkage.
    pub point_correction: f64,
    /// Conditional outcome-noise contribution to the point variance.
    pub noise_variance: f64,
    /// Random-target contribution to the point variance.
    pub target_variance: f64,
    /// Residual degrees of freedom of the independent Gaussian scale estimate.
    pub noise_degrees_of_freedom: usize,
}

/// Generic name for the shared complete-coefficient Gaussian population interval.
pub type PopulationLinearInterval = PopulationAipwInterval;

/// Complete coefficients of a cross-fitted weighted AIPW mean, conditional on design.
/// `folds` and `propensity` must be the actual score-producing nuisance records.
#[allow(clippy::too_many_arguments, reason = "aligned conditional-design arrays")]
pub(crate) fn point_coefficients(
    x: &[f64],
    n: usize,
    p: usize,
    treatment: &[f64],
    weights: &[f64],
    propensity: &[f64],
    folds: &[u32],
    ridge: f64,
) -> Result<Vec<f64>, EstimationError> {
    if x.len() != n * p
        || treatment.len() != n
        || weights.len() != n
        || propensity.len() != n
        || folds.len() != n
        || n == 0
        || p == 0
        || !ridge.is_finite()
        || ridge < 0.0
        || propensity.iter().any(|e| !e.is_finite() || *e <= 0.0 || *e >= 1.0)
        || weights.iter().any(|w| !w.is_finite() || *w <= 0.0)
    {
        return Err(EstimationError::data_msg("invalid linear AIPW design"));
    }
    let total = weights.iter().sum::<f64>();
    let mut h: Vec<f64> = (0..n)
        .map(|i| {
            weights[i] / total
                * if treatment[i] > 0.5 {
                    1.0 / propensity[i]
                } else {
                    -1.0 / (1.0 - propensity[i])
                }
        })
        .collect();
    let mut labels = folds.to_vec();
    labels.sort_unstable();
    labels.dedup();
    for fold in labels {
        for active in [false, true] {
            let rows: Vec<usize> =
                (0..n).filter(|&i| folds[i] != fold && (treatment[i] > 0.5) == active).collect();
            if rows.len() <= p {
                return Err(EstimationError::data_msg("insufficient linear AIPW training arm"));
            }
            let m = rows.len();
            let design: Vec<f64> = (0..p)
                .flat_map(|c| rows.iter().map(move |&i| x[c * n + i] * weights[i].sqrt()))
                .collect();
            // Use the actual learner's penalty/intercept rule after prior-weight scaling.
            let inverse = antecedent_stats::ridge_gram_inverse(&design, m, p, ridge)
                .map_err(crate::util::stats_err)?
                .ok_or_else(|| EstimationError::stats_msg("singular AIPW training design"))?;
            let mut load = vec![0.0; p];
            for i in (0..n).filter(|&i| folds[i] == fold) {
                let u = if active {
                    1.0 - treatment[i] / propensity[i]
                } else {
                    -1.0 + (1.0 - treatment[i]) / (1.0 - propensity[i])
                };
                for c in 0..p {
                    load[c] += weights[i] / total * u * x[c * n + i];
                }
            }
            let beta: Vec<f64> =
                (0..p).map(|c| (0..p).map(|d| inverse[c * p + d] * load[d]).sum()).collect();
            for i in rows {
                h[i] += weights[i] * (0..p).map(|c| x[c * n + i] * beta[c]).sum::<f64>();
            }
        }
    }
    Ok(h)
}

/// Correct conditional arm-linear mean reproduction, leaving residual-space noise intact.
fn correct_mean(
    problem: &PreparedPropensityProblem,
    weights: &[f64],
    h: &mut [f64],
) -> Result<(), EstimationError> {
    let n = problem.nrows;
    let p = problem.design_ncols;
    let total = weights.iter().sum::<f64>();
    let x = &problem.design_matrix;
    for active in [false, true] {
        let mut gram = vec![0.0; p * p];
        let mut rhs = vec![0.0; p];
        for c in 0..p {
            let target = weights.iter().enumerate().map(|(i, w)| w * x[c * n + i]).sum::<f64>()
                / total
                * if active { 1.0 } else { -1.0 };
            rhs[c] = target;
            for i in (0..n).filter(|&i| (problem.treatment[i] > 0.5) == active) {
                rhs[c] -= x[c * n + i] * h[i];
                for d in 0..p {
                    gram[c * p + d] += x[c * n + i] * x[d * n + i];
                }
            }
        }
        let beta = crate::util::solve_spd(&gram, &rhs, p)
            .ok_or_else(|| EstimationError::stats_msg("singular AIPW mean correction"))?;
        for i in (0..n).filter(|&i| (problem.treatment[i] > 0.5) == active) {
            h[i] += (0..p).map(|c| x[c * n + i] * beta[c]).sum::<f64>();
        }
    }
    Ok(())
}

pub(crate) fn interval(
    problem: &PreparedPropensityProblem,
    estimate: EffectEstimate,
    propensity: &[f64],
    folds: &[u32],
    ridge: f64,
    level: f64,
) -> Result<PopulationAipwInterval, EstimationError> {
    if !level.is_finite()
        || level <= 0.0
        || level >= 1.0
        || !matches!(problem.target_population, antecedent_core::TargetPopulation::AllObserved)
        || crate::propensity::trim_of(problem.overlap).is_some()
        || problem.fold_units.is_some()
    {
        return Err(EstimationError::unsupported(
            "population linear AIPW requires independent untrimmed ATE rows and 0 < level < 1",
        ));
    }
    let n = problem.nrows;
    let weights =
        problem.sampling_weights.as_ref().map_or_else(|| vec![1.0; n], |s| s.weights.to_vec());
    let mut h = point_coefficients(
        &problem.design_matrix,
        n,
        problem.design_ncols,
        &problem.treatment,
        &weights,
        propensity,
        folds,
        ridge,
    )?;
    let raw = h.iter().zip(problem.outcome.iter()).map(|(a, b)| a * b).sum::<f64>();
    if (raw - estimate.ate).abs() > 1e-7 * (1.0 + estimate.ate.abs()) {
        return Err(EstimationError::stats_msg(
            "linear AIPW coefficient does not reproduce fitted point",
        ));
    }
    correct_mean(problem, &weights, &mut h)?;
    linear_interval(problem, estimate, &h, level)
}

/// Shared model interval for an outcome-independent coefficient reproducing the
/// weighted arm-linear target. Callers must construct h without consulting outcomes.
pub(crate) fn linear_interval(
    problem: &PreparedPropensityProblem,
    mut estimate: EffectEstimate,
    h: &[f64],
    level: f64,
) -> Result<PopulationLinearInterval, EstimationError> {
    let n = problem.nrows;
    let weights =
        problem.sampling_weights.as_ref().map_or_else(|| vec![1.0; n], |s| s.weights.to_vec());
    if h.len() != n
        || !level.is_finite()
        || level <= 0.0
        || level >= 1.0
        || !matches!(problem.target_population, antecedent_core::TargetPopulation::AllObserved)
        || crate::propensity::trim_of(problem.overlap).is_some()
        || problem.fold_units.is_some()
    {
        return Err(EstimationError::unsupported(
            "population linear interval requires IID untrimmed ATE and valid level",
        ));
    }
    let total = weights.iter().sum::<f64>();
    for active in [false, true] {
        for c in 0..problem.design_ncols {
            let target = (0..n).map(|i| weights[i] * problem.design_matrix[c * n + i]).sum::<f64>()
                / total
                * if active { 1.0 } else { -1.0 };
            let actual = (0..n)
                .filter(|&i| (problem.treatment[i] > 0.5) == active)
                .map(|i| h[i] * problem.design_matrix[c * n + i])
                .sum::<f64>();
            if !actual.is_finite() || (actual - target).abs() > 1e-7 * (1.0 + target.abs()) {
                return Err(EstimationError::stats_msg(
                    "linear interval coefficient does not reproduce arm-linear population mean",
                ));
            }
        }
    }
    let point = h.iter().zip(problem.outcome.iter()).map(|(a, b)| a * b).sum::<f64>();
    let noise = crate::propensity::population_linear_noise(problem, h)?;
    let total = weights.iter().sum::<f64>();
    let target_mean =
        weights.iter().zip(&noise.contrast_mean).map(|(w, t)| w * t).sum::<f64>() / total;
    let mass: Vec<f64> = weights.iter().map(|w| n as f64 * w / total).collect();
    let signal: Vec<f64> =
        mass.iter().zip(&noise.contrast_mean).map(|(w, t)| w * (t - target_mean)).collect();
    let sample_cov = |a: &[f64], b: &[f64]| {
        let ma = a.iter().sum::<f64>() / n as f64;
        let mb = b.iter().sum::<f64>() / n as f64;
        a.iter().zip(b).map(|(a, b)| (a - ma) * (b - mb)).sum::<f64>() / (n * (n - 1)) as f64
    };
    let target_variance = sample_cov(&signal, &signal);
    let mass_variance = sample_cov(&mass, &mass);
    let covariance = sample_cov(&signal, &mass);
    let variance = noise.variance + target_variance;
    let z2 = antecedent_stats::normal_ppf((1.0 + level) / 2.0).powi(2);
    let denominator = 1.0 - z2 * mass_variance;
    if !variance.is_finite() || denominator <= 0.0 {
        return Err(EstimationError::stats_msg("unbounded population AIPW confidence set"));
    }
    let centre = -z2 * covariance / denominator;
    let radius =
        (z2 * z2 * covariance * covariance + denominator * z2 * variance).sqrt() / denominator;
    let correction = point - estimate.ate;
    estimate.ate = point;
    estimate.uncertainty.analytic.se = variance.sqrt();
    estimate.uncertainty.analytic.kind = Some(crate::AnalyticSeKind::Homoskedastic);
    estimate.uncertainty.analytic.reference_df = None;
    // The old score table describes the raw score construction, not the corrected point.
    estimate.scores = crate::EstimateScores::default();
    estimate.joint.covariance = None;
    Ok(PopulationAipwInterval {
        estimate,
        lower: point + centre - radius,
        upper: point + centre + radius,
        level,
        point_correction: correction,
        noise_variance: noise.variance,
        target_variance,
        noise_degrees_of_freedom: noise.degrees_of_freedom,
    })
}

/// Execute population-AIPW prepared requests through the certified nuisance cache.
///
/// # Errors
///
/// [`crate::nuisance_cache::NuisanceCacheError::WrongFamily`] when a request is not AIPW.
#[must_use = "a refused population AIPW subplan is not a certified fit"]
pub fn execute_certified_population_aipw(
    cache: &mut crate::nuisance_cache::NuisanceCache,
    requests: &[crate::nuisance_cache::PreparedFamilyRequest],
    produce: impl FnMut(
        &crate::nuisance_cache::NuisanceCacheKey,
        crate::nuisance_cache::NuisanceRole,
    ) -> Result<
        Vec<crate::nuisance_cache::BlockedOutOfFoldPredictions>,
        crate::nuisance_cache::NuisanceCacheError,
    >,
) -> Result<crate::nuisance_cache::BatchExecution, crate::nuisance_cache::NuisanceCacheError> {
    crate::nuisance_cache::execute_family_requests(
        crate::nuisance_cache::EstimatorFamily::Aipw,
        cache,
        requests,
        produce,
    )
}

/// Restore the certified treatment propensity in fold order.
///
/// # Errors
///
/// Restoration errors of [`crate::nuisance_cache::NuisanceCache::restore_rows`].
#[must_use = "a refused propensity restore is not the certified fit"]
pub fn certified_propensity(
    cache: &crate::nuisance_cache::NuisanceCache,
    key: &crate::nuisance_cache::NuisanceCacheKey,
) -> Result<Vec<f64>, crate::nuisance_cache::NuisanceCacheError> {
    let mut values = Vec::new();
    for fold in 0..key.folds().fold_count() {
        let ids: Vec<u64> = key
            .folds()
            .rows()
            .iter()
            .filter(|row| row.fold() == fold)
            .map(crate::nuisance_cache::FoldRow::row_id)
            .collect();
        if ids.is_empty() {
            continue;
        }
        values.extend(cache.restore_rows(
            key,
            crate::nuisance_cache::NuisanceRole::Treatment,
            fold,
            &ids,
        )?);
    }
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_stats::{FaerBackend, LeastSquaresWorkspace};

    #[test]
    fn population_aipw_calls_the_certified_cache() {
        use std::cell::Cell;

        use crate::nuisance_cache::{
            BlockedOutOfFoldPredictions, CertifiedSource, EstimatorFamily, FoldAssignment,
            FoldRow, NuisanceCache, NuisanceCacheKey, NuisanceColumns, NuisanceInputs,
            NuisanceRole, PredictedRow, PreparedFamilyRequest, RequestColumns,
        };
        use antecedent_core::{NumericalPolicy, SourceDependencyId};
        let folds =
            FoldAssignment::try_new(2, &[FoldRow::new(10, 0), FoldRow::new(40, 1)], &[]).unwrap();
        let outcome = [0.2f64, 0.4];
        let treatment = [0.0f64, 1.0];
        let design = [1.0f64, 1.0, 0.2, 0.5];
        let options = [1.0f64];
        let columns =
            NuisanceColumns::try_new(&outcome, &treatment, &design, &options, None).unwrap();
        let inputs =
            NuisanceInputs::try_new(columns, 6, &folds, "arm", NumericalPolicy::strict()).unwrap();
        let key = NuisanceCacheKey::from_inputs(&inputs).unwrap();
        let prepared = PreparedFamilyRequest::try_new(
            EstimatorFamily::Aipw,
            RequestColumns::try_new("outcome", "arm", &["z"]).unwrap(),
            CertifiedSource::new(SourceDependencyId::from_raw(3), key.outcome_digest()),
            key,
            &[NuisanceRole::Treatment],
        )
        .unwrap();
        let mut cache = NuisanceCache::new();
        let calls = Cell::new(0u64);
        let run = |cache: &mut NuisanceCache, requests: &[PreparedFamilyRequest]| {
            execute_certified_population_aipw(cache, requests, |_, _| {
                calls.set(calls.get().saturating_add(1));
                Ok(vec![
                    BlockedOutOfFoldPredictions::try_new(
                        0,
                        &[PredictedRow::try_new(10, 0.2).unwrap()],
                    )
                    .unwrap(),
                    BlockedOutOfFoldPredictions::try_new(
                        1,
                        &[PredictedRow::try_new(40, 0.4).unwrap()],
                    )
                    .unwrap(),
                ])
            })
        };
        let first = run(&mut cache, &[prepared.clone(), prepared.clone()]).unwrap();
        assert_eq!(calls.get(), 1);
        assert_eq!((first.receipt().scans, first.receipt().fits, first.receipt().hits), (1, 1, 0));
        let restored = certified_propensity(&cache, prepared.key()).unwrap();
        assert_eq!(restored[0].to_bits(), 0.2f64.to_bits());
        assert_eq!(restored[1].to_bits(), 0.4f64.to_bits());
        calls.set(0);
        let second = run(&mut cache, &[prepared]).unwrap();
        assert_eq!(calls.get(), 0);
        assert_eq!(
            (second.receipt().scans, second.receipt().fits, second.receipt().hits),
            (0, 0, 1)
        );
        assert_eq!(first.fits()[0], second.fits()[0]);
    }

    #[test]
    #[allow(
        clippy::needless_range_loop,
        reason = "retain explicit matrix indices in the independent dense outcome-basis oracle"
    )]
    fn training_coefficient_matches_independent_dense_ridge_fit_for_every_outcome_basis() {
        let n = 40;
        let p = 2;
        let mut x = vec![1.0; n * p];
        for i in 0..n {
            x[n + i] = (i as f64 * 0.73).sin();
        }
        let t: Vec<f64> = (0..n).map(|i| (i % 2) as f64).collect();
        let w: Vec<f64> = (0..n).map(|i| 0.5 + (i % 7) as f64 / 4.0).collect();
        let e: Vec<f64> = (0..n).map(|i| 0.25 + (i % 5) as f64 / 10.0).collect();
        let folds: Vec<u32> =
            (0..n).map(|i| u32::try_from((i / 2) % 5).expect("five folds")).collect();
        for ridge in [0.0, 1.0] {
            let h = point_coefficients(&x, n, p, &t, &w, &e, &folds, ridge).unwrap();
            for basis in 0..n {
                let mut predictions = [vec![0.0; n], vec![0.0; n]];
                for fold in 0..5 {
                    for arm in 0..2 {
                        let rows: Vec<usize> = (0..n)
                            .filter(|&i| {
                                folds[i] != fold && t[i].to_bits() == (arm as f64).to_bits()
                            })
                            .collect();
                        let m = rows.len();
                        let design: Vec<f64> = (0..p)
                            .flat_map(|c| {
                                rows.iter().map({
                                    let x = &x;
                                    let w = &w;
                                    move |&i| x[c * n + i] * w[i].sqrt()
                                })
                            })
                            .collect();
                        let y: Vec<f64> = rows
                            .iter()
                            .map(|&i| if i == basis { w[i].sqrt() } else { 0.0 })
                            .collect();
                        let fit = antecedent_stats::fit_ridge(
                            &design,
                            m,
                            p,
                            &y,
                            ridge,
                            &FaerBackend,
                            &mut LeastSquaresWorkspace::default(),
                        )
                        .unwrap();
                        for i in (0..n).filter(|&i| folds[i] == fold) {
                            predictions[arm][i] =
                                fit.coefficients[0] + fit.coefficients[1] * x[n + i];
                        }
                    }
                }
                let point = (0..n)
                    .map(|i| {
                        let y = f64::from(i == basis);
                        w[i] * (predictions[1][i] - predictions[0][i]
                            + t[i] / e[i] * (y - predictions[1][i])
                            - (1.0 - t[i]) / (1.0 - e[i]) * (y - predictions[0][i]))
                    })
                    .sum::<f64>()
                    / w.iter().sum::<f64>();
                assert!(
                    (point - h[basis]).abs() < 1e-11,
                    "ridge={ridge}, basis={basis}: {point} vs {}",
                    h[basis]
                );
            }
        }
    }
}
