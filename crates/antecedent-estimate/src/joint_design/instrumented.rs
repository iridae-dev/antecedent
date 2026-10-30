//! Joint structural-feature instrumental-variable contrasts; no generic LATE-to-ATE conversion.
// SPDX-License-Identifier: MIT OR Apache-2.0
use super::{JointDesignEstimate, JointUncertainty};
use antecedent_stats::{FaerBackend, IvBlocks, LeastSquaresWorkspace, fit_joint_2sls};
use std::collections::BTreeSet;
/// Independent observation with actual endogenous structural features and excluded instruments.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JointInstrumentedUnit {
    /// Original sampling-unit identity.
    pub id: String,
    /// Actual structural features, including actual dose interactions where requested.
    pub endogenous: Vec<f64>,
    /// Excluded instruments for the joint feature space.
    pub instruments: Vec<f64>,
    /// Included exogenous features, with an intercept here if required.
    pub exogenous: Vec<f64>,
    /// Observed outcome.
    pub outcome: f64,
}
/// Explicit homogeneous structural moment model supporting joint feature contrasts.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JointInstrumentedDesign {
    /// Stable target key and endogenous feature contrast (not raw marginal doses).
    pub contrasts: Vec<(String, Vec<f64>)>,
    /// Identified structural equation and feature definitions.
    pub structural_model: String,
    /// Explicit restrictions E[Z ε]=0, including exogeneity and exclusion meaning.
    pub moment_restrictions: String,
    /// Population and independent-unit sampling statement.
    pub population: String,
}
fn coefficients(units: &[&JointInstrumentedUnit]) -> Result<Vec<f64>, String> {
    let n = units.len();
    let p = units[0].endogenous.len();
    let q = units[0].instruments.len();
    let r = units[0].exogenous.len();
    let columns = |get: fn(&JointInstrumentedUnit) -> &Vec<f64>, width: usize| {
        (0..width).flat_map(|j| units.iter().map(move |u| get(u)[j])).collect::<Vec<_>>()
    };
    let endogenous = columns(|u| &u.endogenous, p);
    let instruments = columns(|u| &u.instruments, q);
    let exogenous = columns(|u| &u.exogenous, r);
    let y = units.iter().map(|u| u.outcome).collect::<Vec<_>>();
    let fit = fit_joint_2sls(
        &IvBlocks {
            nrows: n,
            instruments_colmajor: &instruments,
            z_ncols: q,
            exogenous_colmajor: &exogenous,
            x_ncols: r,
        },
        &endogenous,
        p,
        &y,
        &FaerBackend,
        &mut LeastSquaresWorkspace::default(),
    )
    .map_err(|e| e.to_string())?;
    Ok(fit.second_stage.coefficients)
}
fn point(
    design: &JointInstrumentedDesign,
    units: &[&JointInstrumentedUnit],
) -> Result<Vec<f64>, String> {
    let beta = coefficients(units)?;
    Ok(design
        .contrasts
        .iter()
        .map(|(_, contrast)| contrast.iter().zip(&beta).map(|(a, b)| a * b).sum())
        .collect())
}
/// Execute joint-feature 2SLS with full instrument and joint projected-design rank checks.
/// Distinct local/complier effects under weaker assumptions are not accepted as
/// premises for this structural-model target. The population is exactly the declared
/// moment-model population. Weak identification remains a statistical caveat; the
/// returned joint delete-unit covariance is uncalibrated and is not weak-IV robust.
///
/// # Errors
/// Missing moment/model premises, invalid dimensions, insufficient joint instrument
/// rank or a full-procedure uncertainty realization losing rank.
pub fn instrumented_joint_contrasts(
    design: &JointInstrumentedDesign,
    units: &[JointInstrumentedUnit],
    ctx: &antecedent_core::ExecutionContext,
) -> Result<JointDesignEstimate, String> {
    if units.len() < 4
        || design.contrasts.is_empty()
        || [&design.structural_model, &design.moment_restrictions, &design.population]
            .iter()
            .any(|s| s.trim().is_empty())
    {
        return Err("joint IV requires explicit structural moments, population and supported feature contrasts".into());
    }
    let p = units[0].endogenous.len();
    let q = units[0].instruments.len();
    let r = units[0].exogenous.len();
    let mut ids = BTreeSet::new();
    if p == 0
        || q < p
        || design
            .contrasts
            .iter()
            .any(|(key, c)| key.is_empty() || c.len() != p || c.iter().any(|v| !v.is_finite()))
        || design.contrasts.iter().map(|(key, _)| key).collect::<BTreeSet<_>>().len()
            != design.contrasts.len()
    {
        return Err("joint IV feature contrasts or excluded instrument dimension invalid".into());
    }
    for u in units {
        if u.id.is_empty()
            || !ids.insert(&u.id)
            || u.endogenous.len() != p
            || u.instruments.len() != q
            || u.exogenous.len() != r
            || !u.outcome.is_finite()
            || u.endogenous.iter().chain(&u.instruments).chain(&u.exogenous).any(|v| !v.is_finite())
        {
            return Err("invalid joint IV independent-unit structural data".into());
        }
    }
    let refs = units.iter().collect::<Vec<_>>();
    let means = point(design, &refs)?;
    let n = units.len();
    let k = means.len();
    // One retained realization per deleted unit, plus the k-by-k covariance.
    let _lease = super::budget::admit_cells(
        ctx,
        "structural IV delete-unit realizations",
        n.saturating_mul(k).saturating_add(k.saturating_mul(k)),
        1,
    )?;
    let mut deletes = Vec::with_capacity(n);
    for omitted in 0..n {
        super::budget::check_cancelled(ctx)?;
        deletes.push(point(
            design,
            &refs
                .iter()
                .enumerate()
                .filter_map(|(i, u)| (i != omitted).then_some(*u))
                .collect::<Vec<_>>(),
        )?);
    }
    let center =
        (0..k).map(|j| deletes.iter().map(|v| v[j]).sum::<f64>() / n as f64).collect::<Vec<_>>();
    let mut covariance = vec![0.0; k * k];
    for v in deletes {
        for j in 0..k {
            for l in 0..k {
                covariance[j * k + l] +=
                    (n - 1) as f64 / n as f64 * (v[j] - center[j]) * (v[l] - center[l]);
            }
        }
    }
    Ok(JointDesignEstimate {
        worlds: design.contrasts.iter().map(|(key, _)| key.clone()).collect(),
        means,
        covariance,
        sampling_unit: "independent IV unit".into(),
        uncertainty: JointUncertainty::DeleteEpisodeJackknife,
    })
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn world_mean_jackknife_includes_source_covariate_sampling_and_legacy_refuses() {
        let mut units = Vec::new();
        for rep in 0..4 {
            for z in [-1.0, 1.0] {
                for u in [-1.0, 1.0] {
                    let x = rep as f64 - 1.5;
                    let dose = z + 0.25 * u;
                    units.push(JointInstrumentedUnit {
                        id: format!("{rep}/{z}/{u}"),
                        endogenous: vec![dose],
                        instruments: vec![z],
                        exogenous: vec![1.0, x],
                        outcome: 2.0 * dose + x + u,
                    });
                }
            }
        }
        let design = JointInstrumentedDesign {
            contrasts: vec![("dose".into(), vec![1.0])],
            structural_model: "linear structural model".into(),
            moment_restrictions: "independent instrument".into(),
            population: "source".into(),
        };
        let features =
            JointStructuralFeatureBinding { raw_feature_indices: vec![0], powers: vec![vec![1]] };
        let fit = fit_instrumented_joint_polynomial(
            &design,
            &units,
            features,
            &antecedent_core::ExecutionContext::for_tests(3),
        )
        .unwrap();
        let (mean, variance) = fit.world_mean(&[0.0]).unwrap();
        let mut values = Vec::new();
        for omitted in 0..units.len() {
            let kept: Vec<_> =
                units.iter().enumerate().filter_map(|(i, u)| (i != omitted).then_some(u)).collect();
            let coefficients = coefficients(&kept).unwrap();
            let xmean = kept.iter().map(|u| u.exogenous[1]).sum::<f64>() / kept.len() as f64;
            values.push(coefficients[1] + coefficients[2] * xmean);
        }
        let center = values.iter().sum::<f64>() / values.len() as f64;
        let reference = (values.len() - 1) as f64 / values.len() as f64
            * values.iter().map(|v| (v - center).powi(2)).sum::<f64>();
        assert!(mean.abs() < 1e-10);
        assert!(reference > 0.0);
        assert!((variance - reference).abs() < 1e-10);
        let encoded = serde_json::to_value(&fit).unwrap();
        let restored: JointInstrumentedModelFit = serde_json::from_value(encoded.clone()).unwrap();
        assert_eq!(fit.world_mean(&[0.0]).unwrap(), restored.world_mean(&[0.0]).unwrap());
        let mut legacy = encoded;
        legacy.as_object_mut().unwrap().remove("world_mean_covariance");
        let old: JointInstrumentedModelFit = serde_json::from_value(legacy).unwrap();
        assert!(old.world_mean(&[0.0]).is_err());
        assert_eq!(old.covariance, fit.covariance);
    }

    #[test]
    fn instruments_identify_actual_joint_features_and_refuse_joint_rank_loss() {
        let mut units = Vec::new();
        for replicate in 0..3 {
            for a in [-1.0, 1.0] {
                for b in [-1.0, 1.0] {
                    for u in [-1.0, 1.0] {
                        let d1 = a + u;
                        let d2 = b + 0.5 * u;
                        let features = vec![d1, d2, d1 * d2];
                        units.push(JointInstrumentedUnit {
                            id: format!("{replicate}/{a}/{b}/{u}"),
                            endogenous: features.clone(),
                            instruments: vec![a, b, a * b],
                            exogenous: vec![1.0],
                            outcome: 2.0 * d1 + 3.0 * d2 + 4.0 * d1 * d2 + 5.0 + u,
                        });
                    }
                }
            }
        }
        let design = JointInstrumentedDesign {
            contrasts: vec![
                ("dose-a".into(), vec![1.0, 0.0, 0.0]),
                ("interaction".into(), vec![0.0, 0.0, 1.0]),
                ("joint".into(), vec![1.0, 1.0, 1.0]),
            ],
            structural_model: "Y=2D1+3D2+4D1D2+intercept+u".into(),
            moment_restrictions: "Z=(A,B,AB) independent u; structural exclusion".into(),
            population: "independent source units under common structural coefficients".into(),
        };
        let fit = instrumented_joint_contrasts(
            &design,
            &units,
            &antecedent_core::ExecutionContext::for_tests(1),
        )
        .unwrap();
        for (got, want) in fit.means.iter().zip([2.0, 4.0, 9.0]) {
            assert!((got - want).abs() < 1e-9);
        }
        let mut invalid = design.clone();
        invalid.moment_restrictions.clear();
        assert!(
            instrumented_joint_contrasts(
                &invalid,
                &units,
                &antecedent_core::ExecutionContext::for_tests(1)
            )
            .is_err()
        );
        for unit in &mut units {
            unit.instruments[2] = unit.instruments[0];
        }
        assert!(
            instrumented_joint_contrasts(
                &design,
                &units,
                &antecedent_core::ExecutionContext::for_tests(1)
            )
            .is_err()
        );
    }
}

/// Proper triangular likelihood on *raw doses* and the structural outcome.
/// Nonlinear endogenous features are deterministic monomials of these doses,
/// not incorrectly assumed jointly Gaussian with their own products.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JointIvGaussianPrior {
    /// Positions of the raw dose columns within the actual endogenous feature vector.
    pub raw_feature_indices: Vec<usize>,
    /// One monomial exponent vector per structural feature, in raw-dose order.
    pub structural_features: Vec<Vec<u32>>,
    /// Proper prior means: [structural beta, included gamma, first-stage coefficients
    /// for raw dose zero, raw dose one, ...]. Each first stage is [instruments, exogenous].
    pub parameter_mean: Vec<f64>,
    /// Proper independent prior variances in the same full parameter order.
    pub parameter_variance: Vec<f64>,
    /// Known row-major covariance of [raw first-stage disturbances, structural outcome
    /// disturbance]. Nonzero cross-covariances explicitly model endogenous doses.
    pub disturbance_covariance: Vec<f64>,
    /// Aligned mechanism draws.
    pub draws: usize,
    /// Deterministic stream seed.
    pub seed: u64,
}
/// Check the proper prior, actual raw-dose feature bindings and declared joint
/// disturbance covariance before forming the triangular likelihood precision.
fn validate_structural_likelihood(
    prior: &JointIvGaussianPrior,
    units: &[JointInstrumentedUnit],
    p: usize,
    parameters: usize,
    errors: usize,
) -> Result<(), String> {
    let raw = prior.raw_feature_indices.len();
    if raw == 0
        || prior.raw_feature_indices.iter().any(|j| *j >= p)
        || prior.raw_feature_indices.iter().collect::<BTreeSet<_>>().len() != raw
        || prior.structural_features.len() != p
        || prior
            .structural_features
            .iter()
            .any(|powers| powers.len() != raw || powers.iter().any(|power| *power > 8))
        || prior.parameter_mean.len() != parameters
        || prior.parameter_variance.len() != parameters
        || prior.parameter_mean.iter().any(|x| !x.is_finite())
        || prior.parameter_variance.iter().any(|x| !x.is_finite() || *x <= 0.0)
        || prior.disturbance_covariance.len() != errors * errors
        || prior.disturbance_covariance.iter().any(|x| !x.is_finite())
        || prior.draws < 2
    {
        return Err("invalid proper joint triangular IV prior or raw-feature specification".into());
    }
    for unit in units {
        for (j, powers) in prior.structural_features.iter().enumerate() {
            let value = powers
                .iter()
                .zip(&prior.raw_feature_indices)
                .map(|(power, raw)| unit.endogenous[*raw].powi(i32::try_from(*power).unwrap_or(9)))
                .product::<f64>();
            if !value.is_finite()
                || (value - unit.endogenous[j]).abs() > 1e-10 * (1.0 + value.abs())
            {
                return Err("actual endogenous feature does not equal the declared raw-dose structural polynomial".into());
            }
        }
    }
    if (0..errors).any(|j| {
        (0..errors).any(|l| {
            (prior.disturbance_covariance[j * errors + l]
                - prior.disturbance_covariance[l * errors + j])
                .abs()
                > 1e-12
        })
    }) {
        return Err("joint triangular IV disturbance covariance must be symmetric".into());
    }
    Ok(())
}

/// Genuine Bayesian joint structural-feature IV likelihood update.
/// The raw first stages and structural equation are fitted jointly with a proper
/// Gaussian parameter prior and declared known correlated Gaussian disturbances.
/// Each structural-feature value is checked against its raw-dose polynomial;
/// projecting raw doses and multiplying projections is never substituted.
/// # Errors
/// Missing native IV premises/rank, invalid feature expansion, nonsingular error
/// covariance/proper priors, excessive dimensions or numerical posterior failure.
pub fn instrumented_joint_coefficient_posterior(
    design: &JointInstrumentedDesign,
    units: &[JointInstrumentedUnit],
    prior: &JointIvGaussianPrior,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<Vec<Vec<f64>>, String> {
    let _identified = instrumented_joint_contrasts(design, units, ctx)?;
    let p = units[0].endogenous.len();
    let q = units[0].instruments.len();
    let r = units[0].exogenous.len();
    let raw = prior.raw_feature_indices.len();
    let stage = q.checked_add(r).ok_or("joint IV first-stage dimension overflow")?;
    let parameters = raw
        .checked_mul(stage)
        .and_then(|v| v.checked_add(p + r))
        .ok_or("joint IV posterior dimension overflow")?;
    let errors = raw + 1;
    validate_structural_likelihood(prior, units, p, parameters, errors)?;
    // The dense precision, its Cholesky factor and the posterior draws are retained together.
    let _lease = super::budget::admit_cells(
        ctx,
        "structural IV posterior",
        parameters
            .saturating_mul(parameters)
            .saturating_mul(2)
            .saturating_add(prior.draws.saturating_mul(p + r)),
        1,
    )?;
    let noise_chol = antecedent_stats::cholesky_spd(&prior.disturbance_covariance, errors)
        .ok_or("joint IV disturbance covariance is not positive definite")?;
    let mut noise_precision = vec![0.0; errors * errors];
    for j in 0..errors {
        let mut e = vec![0.0; errors];
        e[j] = 1.0;
        let inverse = antecedent_stats::chol_solve(&noise_chol, errors, &e)
            .ok_or("joint IV noise precision solve failed")?;
        for l in 0..errors {
            noise_precision[l * errors + j] = inverse[l];
        }
    }
    let mut precision = vec![0.0; parameters * parameters];
    let mut rhs = vec![0.0; parameters];
    for j in 0..parameters {
        precision[j * parameters + j] = 1.0 / prior.parameter_variance[j];
        rhs[j] = prior.parameter_mean[j] / prior.parameter_variance[j];
    }
    for unit in units {
        let mut rows = vec![Vec::<(usize, f64)>::new(); errors];
        for (a, row) in rows.iter_mut().take(raw).enumerate() {
            let offset = p + r + a * stage;
            row.extend(
                unit.instruments
                    .iter()
                    .chain(&unit.exogenous)
                    .enumerate()
                    .map(|(j, v)| (offset + j, *v)),
            );
        }
        rows[raw].extend(
            unit.endogenous.iter().chain(&unit.exogenous).enumerate().map(|(j, v)| (j, *v)),
        );
        let observation = prior
            .raw_feature_indices
            .iter()
            .map(|j| unit.endogenous[*j])
            .chain(std::iter::once(unit.outcome))
            .collect::<Vec<_>>();
        for a in 0..errors {
            for b in 0..errors {
                let weight = noise_precision[a * errors + b];
                for &(j, x) in &rows[a] {
                    rhs[j] += x * weight * observation[b];
                    for &(l, y) in &rows[b] {
                        precision[j * parameters + l] += x * weight * y;
                    }
                }
            }
        }
    }
    let chol = antecedent_stats::cholesky_spd(&precision, parameters)
        .ok_or("joint IV posterior precision is not positive definite")?;
    let mean = antecedent_stats::chol_solve(&chol, parameters, &rhs)
        .ok_or("joint IV posterior coefficient solve failed")?;
    // Draw the full conditional mechanism. Solving L^T delta=z avoids explicitly
    // inverting a large posterior covariance and retains first-stage uncertainty.
    let mut rng = antecedent_core::CausalRng::from_seed(prior.seed);
    let mut draws = Vec::with_capacity(prior.draws);
    for _ in 0..prior.draws {
        super::budget::check_cancelled(ctx)?;
        let mut delta = (0..parameters)
            .map(|_| antecedent_kernels::standard_normal(&mut rng))
            .collect::<Vec<_>>();
        for j in (0..parameters).rev() {
            let sum = (j + 1..parameters).map(|l| chol[l * parameters + j] * delta[l]).sum::<f64>();
            delta[j] = (delta[j] - sum) / chol[j * parameters + j];
        }
        draws.push((0..p + r).map(|j| mean[j] + delta[j]).collect());
    }
    Ok(draws)
}

/// Project genuine joint structural/first-stage posterior draws onto the retained IV contrasts.
/// # Errors
/// Same design, likelihood and feature-binding refusals as the full coefficient owner.
pub fn instrumented_joint_posterior(
    design: &JointInstrumentedDesign,
    units: &[JointInstrumentedUnit],
    prior: &JointIvGaussianPrior,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<super::JointDesignPosterior, String> {
    let coefficients = instrumented_joint_coefficient_posterior(design, units, prior, ctx)?;
    let draws = coefficients
        .iter()
        .map(|beta| {
            design
                .contrasts
                .iter()
                .map(|(_, contrast)| contrast.iter().zip(beta).map(|(c, b)| c * b).sum())
                .collect()
        })
        .collect();
    super::JointDesignPosterior::from_draws(design.contrasts.iter().map(|(key,_)|key.clone()).collect(),draws,"independent IV source unit".into(),"proper joint Gaussian triangular raw-dose/structural-outcome likelihood with declared known correlated disturbances, deterministic verified polynomial structural features and proper Gaussian priors on every structural and first-stage coefficient; explicit homogeneous structural moment target, not generic local/complier population conversion".into())
}

/// A verified polynomial expansion of the recorded raw endogenous doses.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JointStructuralFeatureBinding {
    /// Raw-dose positions within each observed endogenous feature vector.
    pub raw_feature_indices: Vec<usize>,
    /// Monomial exponent vector per structural feature, in raw-dose order.
    pub powers: Vec<Vec<u32>>,
}
/// Fitted homogeneous joint structural response, with full coefficient uncertainty.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct JointInstrumentedModelFit {
    /// Endogenous-feature coefficients followed by included exogenous coefficients.
    pub coefficients: Vec<f64>,
    /// Full row-major coefficient covariance from complete independent-unit deletion.
    pub covariance: Vec<f64>,
    /// Delete-unit covariance of endogenous coefficients and the exogenous population mean contribution.
    /// Absent on legacy serialized fits, which cannot quantify a population world mean.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub world_mean_covariance: Option<Vec<f64>>,
    /// Native structural feature expansion checked against every original observation.
    pub features: JointStructuralFeatureBinding,
    /// Included exogenous feature count.
    pub exogenous_columns: usize,
    /// Mean of each original source exogenous feature. These same population features
    /// are retained under raw-dose interventions, rather than supplied transport values.
    pub source_exogenous_means: Vec<f64>,
    /// Original structural equation and homogeneous moment-model premises.
    pub design: JointInstrumentedDesign,
}
fn verify_features(
    units: &[JointInstrumentedUnit],
    features: &JointStructuralFeatureBinding,
) -> Result<(), String> {
    let p = units.first().ok_or("structural model needs original observations")?.endogenous.len();
    let raw = features.raw_feature_indices.len();
    if raw == 0
        || features.raw_feature_indices.iter().any(|j| *j >= p)
        || features.raw_feature_indices.iter().collect::<BTreeSet<_>>().len() != raw
        || features.powers.len() != p
        || features
            .powers
            .iter()
            .any(|powers| powers.len() != raw || powers.iter().any(|power| *power > 8))
    {
        return Err("invalid structural raw-dose polynomial binding".into());
    }
    for unit in units {
        for (j, powers) in features.powers.iter().enumerate() {
            let value = powers
                .iter()
                .zip(&features.raw_feature_indices)
                .map(|(power, index)| {
                    unit.endogenous[*index].powi(i32::try_from(*power).unwrap_or(9))
                })
                .product::<f64>();
            if !value.is_finite()
                || (value - unit.endogenous[j]).abs() > 1e-10 * (1.0 + value.abs())
            {
                return Err(
                    "actual endogenous feature does not equal its retained raw-dose polynomial"
                        .into(),
                );
            }
        }
    }
    Ok(())
}
/// Fit a joint structural response that may be evaluated at actual raw-dose programs.
/// Intercept inclusion establishes the unconditional zero-mean disturbance moment.
/// The original population's exogenous feature distribution is retained in every
/// world; arbitrary new covariate populations require a separate supported translation.
/// # Errors
/// Missing moment/rank premises, absent intercept, mismatched raw-dose feature binding
/// or a complete-unit uncertainty realization losing rank.
pub fn fit_instrumented_joint_polynomial(
    design: &JointInstrumentedDesign,
    units: &[JointInstrumentedUnit],
    features: JointStructuralFeatureBinding,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<JointInstrumentedModelFit, String> {
    let _identified = instrumented_joint_contrasts(design, units, ctx)?;
    verify_features(units, &features)?;
    let p = units[0].endogenous.len();
    let r = units[0].exogenous.len();
    #[allow(
        clippy::float_cmp,
        reason = "the unconditional disturbance moment requires an exactly coded intercept column of ones, not an approximately constant covariate"
    )]
    let has_intercept = (0..r).any(|j| units.iter().all(|u| u.exogenous[j] == 1.0));
    if !has_intercept {
        return Err("population structural world means require an included intercept moment establishing E[disturbance]=0".into());
    }
    let refs = units.iter().collect::<Vec<_>>();
    let point = coefficients(&refs)?;
    let n = units.len();
    let k = p + r;
    let _lease = super::budget::admit_cells(
        ctx,
        "structural IV delete-unit realizations",
        n.saturating_mul(k.saturating_add(p).saturating_add(1))
            .saturating_add(k.saturating_mul(k))
            .saturating_add((p + 1).saturating_mul(p + 1)),
        1,
    )?;
    let exogenous_sums: Vec<f64> =
        (0..r).map(|j| units.iter().map(|u| u.exogenous[j]).sum()).collect();
    let mut deletes = Vec::with_capacity(n);
    let mut world_deletes = Vec::with_capacity(n);
    for omitted in 0..n {
        super::budget::check_cancelled(ctx)?;
        let beta = coefficients(
            &refs
                .iter()
                .enumerate()
                .filter_map(|(i, u)| (i != omitted).then_some(*u))
                .collect::<Vec<_>>(),
        )?;
        let mut world = beta[..p].to_vec();
        let mean_contribution = (0..r)
            .map(|j| {
                beta[p + j] * (exogenous_sums[j] - units[omitted].exogenous[j]) / (n - 1) as f64
            })
            .sum();
        world.push(mean_contribution);
        world_deletes.push(world);
        deletes.push(beta);
    }
    let world_width = p + 1;
    let world_center: Vec<f64> = (0..world_width)
        .map(|j| world_deletes.iter().map(|v| v[j]).sum::<f64>() / n as f64)
        .collect();
    let mut world_covariance = vec![0.0; world_width * world_width];
    for world in world_deletes {
        for j in 0..world_width {
            for l in 0..world_width {
                world_covariance[j * world_width + l] += (n - 1) as f64 / n as f64
                    * (world[j] - world_center[j])
                    * (world[l] - world_center[l]);
            }
        }
    }
    let center =
        (0..k).map(|j| deletes.iter().map(|v| v[j]).sum::<f64>() / n as f64).collect::<Vec<_>>();
    let mut covariance = vec![0.0; k * k];
    for v in deletes {
        for j in 0..k {
            for l in 0..k {
                covariance[j * k + l] +=
                    (n - 1) as f64 / n as f64 * (v[j] - center[j]) * (v[l] - center[l]);
            }
        }
    }
    let source_exogenous_means =
        (0..r).map(|j| units.iter().map(|u| u.exogenous[j]).sum::<f64>() / n as f64).collect();
    Ok(JointInstrumentedModelFit {
        coefficients: point,
        covariance,
        world_mean_covariance: Some(world_covariance),
        features,
        exogenous_columns: r,
        source_exogenous_means,
        design: design.clone(),
    })
}
impl JointInstrumentedModelFit {
    /// Evaluate a raw-dose world in the original source population.
    /// # Errors
    /// Invalid raw-dose axis, nonfinite feature/mean or coefficient covariance.
    pub fn world_mean(&self, raw_doses: &[f64]) -> Result<(f64, f64), String> {
        if raw_doses.len() != self.features.raw_feature_indices.len()
            || raw_doses.iter().any(|v| !v.is_finite())
        {
            return Err("structural world needs every finite raw-dose control".into());
        }
        let mut row = self
            .features
            .powers
            .iter()
            .map(|powers| {
                powers
                    .iter()
                    .zip(raw_doses)
                    .map(|(power, d)| d.powi(i32::try_from(*power).unwrap_or(9)))
                    .product::<f64>()
            })
            .collect::<Vec<_>>();
        row.extend_from_slice(&self.source_exogenous_means);
        let mean = row.iter().zip(&self.coefficients).map(|(x, b)| x * b).sum::<f64>();
        let p = self.features.powers.len();
        let width = p + 1;
        let covariance = self
            .world_mean_covariance
            .as_ref()
            .ok_or("world-mean uncertainty is unavailable on a legacy structural fit")?;
        if covariance.len() != width.saturating_mul(width)
            || covariance.iter().any(|v| !v.is_finite())
        {
            return Err("invalid world-mean covariance".into());
        }
        let mut world_row = row[..p].to_vec();
        world_row.push(1.0);
        let variance = (0..width)
            .flat_map(|j| (0..width).map(move |l| (j, l)))
            .map(|(j, l)| world_row[j] * world_row[l] * covariance[j * width + l])
            .sum::<f64>();
        if !mean.is_finite() || !variance.is_finite() || variance < -1e-10 {
            return Err("structural world arithmetic overflow or invalid covariance".into());
        }
        Ok((mean, variance.max(0.0)))
    }
    /// Compile the actual fitted polynomial, with only original-population exogenous means.
    /// Caller monotonicity/concavity declarations cannot alter this expression.
    /// # Errors
    /// Repeated/incorrect raw-control axis or an invalid coefficient mapping.
    pub fn expression(
        &self,
        raw_controls: &[antecedent_core::VariableId],
    ) -> Result<antecedent_core::ValueExpr, String> {
        use antecedent_core::ValueExpr;
        if raw_controls.len() != self.features.raw_feature_indices.len()
            || raw_controls.iter().collect::<BTreeSet<_>>().len() != raw_controls.len()
        {
            return Err("structural expression requires a unique complete raw-control axis".into());
        }
        let mut expression = ValueExpr::constant(
            self.source_exogenous_means
                .iter()
                .zip(&self.coefficients[self.features.powers.len()..])
                .map(|(x, b)| x * b)
                .sum::<f64>(),
        );
        for (powers, beta) in self.features.powers.iter().zip(&self.coefficients) {
            let mut term = ValueExpr::constant(*beta);
            for (&variable, &power) in raw_controls.iter().zip(powers) {
                for _ in 0..power {
                    term = term.mul(ValueExpr::var(variable));
                }
            }
            expression = expression.add(term);
        }
        Ok(expression)
    }
}
