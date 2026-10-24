//! Whole-program randomization over independently assigned units or clusters.
// SPDX-License-Identifier: MIT OR Apache-2.0
use super::{JointDesignEstimate, JointUncertainty};
use std::collections::BTreeSet;

/// One independently randomized unit, which may itself be a geographical cluster.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JointRandomizedUnit {
    /// Original observational identity, preserved by held-out evaluation.
    pub id: String,
    /// Index of assigned complete multichannel program.
    pub arm: usize,
    /// Known assignment probabilities for every complete program.
    pub probabilities: Vec<f64>,
    /// Outcome mean or aggregate in the declared population.
    pub outcome: f64,
    /// Fixed preassignment population weight, never a post-treatment sample size.
    pub population_weight: f64,
}
/// Required causal premises for known whole-program assignment.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JointRandomizedDesign {
    /// Complete programs, including full schedules and prehistory in the caller's binding.
    pub worlds: Vec<String>,
    /// Declared independent assignment unit; e.g. participant or geographical cluster.
    pub sampling_unit: String,
    /// Explicit target population description.
    pub population: String,
    /// Scientific consistency statement, including treatment version and measurement.
    pub consistency: String,
    /// No between-unit interference statement, or isolation provided by clustering.
    pub between_unit_isolation: String,
}
/// Horvitz--Thompson complete-program means with fixed population weighting.
/// Units may have unequal known arm probabilities. Cluster-level scores retain
/// joint covariance; no individual-level pseudoreplication enters a geo experiment.
/// The same owner evaluates a prespecified selected program on genuinely held-out units.
///
/// # Errors
/// Missing causal premises, duplicate units/worlds, malformed probabilities or unsupported arms.
pub fn randomized_joint_means(
    design: &JointRandomizedDesign,
    units: &[JointRandomizedUnit],
) -> Result<JointDesignEstimate, String> {
    let k = design.worlds.len();
    if k < 2
        || k > 256
        || units.len() > 65536
        || units.len() < 2
        || [
            &design.sampling_unit,
            &design.population,
            &design.consistency,
            &design.between_unit_isolation,
        ]
        .iter()
        .any(|s| s.trim().is_empty())
        || design.worlds.iter().any(|s| s.is_empty())
        || design.worlds.iter().collect::<BTreeSet<_>>().len() != k
    {
        return Err("joint randomization requires distinct programs, independent units and explicit population/consistency/isolation premises".into());
    }
    let mut ids = BTreeSet::new();
    let mut total = 0.0;
    let mut scores = Vec::with_capacity(units.len());
    let mut observed = vec![0usize; k];
    for unit in units {
        if unit.id.is_empty()
            || !ids.insert(&unit.id)
            || unit.arm >= k
            || unit.probabilities.len() != k
            || unit.probabilities.iter().any(|p| !p.is_finite() || *p <= 0.0 || *p > 1.0)
            || (unit.probabilities.iter().sum::<f64>() - 1.0).abs() > 1e-12
            || !unit.outcome.is_finite()
            || !unit.population_weight.is_finite()
            || unit.population_weight <= 0.0
        {
            return Err(
                "invalid joint arm assignment, population weight or original unit identity".into(),
            );
        }
        total += unit.population_weight;
        let mut score = vec![0.0; k];
        score[unit.arm] = unit.population_weight * unit.outcome / unit.probabilities[unit.arm];
        observed[unit.arm] += 1;
        scores.push(score);
    }
    if observed.contains(&0) {
        return Err("requested complete program has no observed independent-unit support".into());
    }
    let means =
        (0..k).map(|j| scores.iter().map(|s| s[j]).sum::<f64>() / total).collect::<Vec<_>>();
    let n = units.len() as f64;
    let mut covariance = vec![0.0; k * k];
    for (unit, score) in units.iter().zip(scores) {
        for j in 0..k {
            for l in 0..k {
                covariance[j * k + l] += n / (n - 1.0)
                    * (score[j] - unit.population_weight * means[j])
                    * (score[l] - unit.population_weight * means[l])
                    / total.powi(2);
            }
        }
    }
    if means.iter().chain(&covariance).any(|v| !v.is_finite()) {
        return Err("joint randomization arithmetic overflow".into());
    }
    Ok(JointDesignEstimate {
        worlds: design.worlds.clone(),
        means,
        covariance,
        sampling_unit: design.sampling_unit.clone(),
        uncertainty: JointUncertainty::IndependentUnitSandwich,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn factorial_interaction_and_geo_population_use_actual_joint_cells() {
        let d = JointRandomizedDesign {
            worlds: vec!["00".into(), "10".into(), "01".into(), "11".into()],
            sampling_unit: "geo".into(),
            population: "fixed census".into(),
            consistency: "complete program including history".into(),
            between_unit_isolation: "no spillovers across geos".into(),
        };
        let mut units = Vec::new();
        for repeat in 0..4 {
            for arm in 0..4 {
                units.push(JointRandomizedUnit {
                    id: format!("{repeat}/{arm}"),
                    arm,
                    probabilities: vec![0.25; 4],
                    outcome: [2.0, 5.0, 7.0, 14.0][arm],
                    population_weight: if repeat < 2 { 1.0 } else { 3.0 },
                });
            }
        }
        let fit = randomized_joint_means(&d, &units).unwrap();
        assert_eq!(fit.means, [2.0, 5.0, 7.0, 14.0]);
        assert_eq!(fit.contrast(&[1.0, -1.0, -1.0, 1.0]).unwrap().0, 4.0);
        let mut invalid = d.clone();
        invalid.between_unit_isolation.clear();
        assert!(randomized_joint_means(&invalid, &units).is_err());
        units[0].probabilities[3] = 0.0;
        assert!(randomized_joint_means(&d, &units).is_err());
    }
}

/// Explicit known-variance Gaussian arm-mean likelihood and proper independent priors.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RandomizedGaussianPrior {
    /// Prior mean for each complete program's population response.
    pub mean: Vec<f64>,
    /// Positive prior variance for each complete program.
    pub variance: Vec<f64>,
    /// Known sampling variance for each independent unit's outcome, in unit order.
    pub outcome_variance: Vec<f64>,
    /// Number of aligned posterior mechanism draws.
    pub draws: usize,
    /// Deterministic sampling seed.
    pub seed: u64,
}
/// Bayesian complete-program response under a declared Gaussian arm-mean likelihood.
/// Known randomization identifies the same arm means as the frequentist owner;
/// the Gaussian likelihood and proper priors are additional statistical assumptions.
/// Fixed population weights are allowed only when equal: unequal weighted population
/// means require a population-response model and cannot be silently fitted as common means.
///
/// # Errors
/// Invalid randomization/premises, likelihood/prior shapes or unequal population weights.
pub fn randomized_joint_posterior(
    design: &JointRandomizedDesign,
    units: &[JointRandomizedUnit],
    prior: &RandomizedGaussianPrior,
) -> Result<super::JointDesignPosterior, String> {
    let _design_check = randomized_joint_means(design, units)?;
    let k = design.worlds.len();
    if prior.mean.len() != k
        || prior.variance.len() != k
        || prior.outcome_variance.len() != units.len()
        || prior.draws < 2
        || prior.draws > 1_000_000
        || prior.mean.iter().any(|v| !v.is_finite())
        || prior.variance.iter().chain(&prior.outcome_variance).any(|v| !v.is_finite() || *v <= 0.0)
        || units.iter().any(|u| u.population_weight != units[0].population_weight)
    {
        return Err("Gaussian randomized posterior requires proper finite priors, known variances and equal population weights".into());
    }
    let mut precision = prior.variance.iter().map(|v| 1.0 / v).collect::<Vec<_>>();
    let mut weighted = prior.mean.iter().zip(&precision).map(|(m, p)| m * p).collect::<Vec<_>>();
    for (u, var) in units.iter().zip(&prior.outcome_variance) {
        precision[u.arm] += 1.0 / var;
        weighted[u.arm] += u.outcome / var;
    }
    let mut rng = antecedent_core::CausalRng::from_seed(prior.seed);
    let draws = (0..prior.draws)
        .map(|_| {
            (0..k)
                .map(|j| {
                    weighted[j] / precision[j]
                        + antecedent_kernels::standard_normal(&mut rng) / precision[j].sqrt()
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    super::JointDesignPosterior::from_draws(design.worlds.clone(),draws,design.sampling_unit.clone(),"independent Gaussian arm-mean likelihood with declared known per-unit outcome variances and proper independent Gaussian arm priors; equal-weight population".into())
}

#[cfg(test)]
mod posterior_tests {
    use super::*;
    #[test]
    fn gaussian_cell_posterior_matches_independent_conjugate_reference() {
        let design = JointRandomizedDesign {
            worlds: vec!["base".into(), "joint".into()],
            sampling_unit: "participant".into(),
            population: "same source population".into(),
            consistency: "assigned joint programs".into(),
            between_unit_isolation: "independent participants".into(),
        };
        let units = (0..20)
            .map(|i| JointRandomizedUnit {
                id: i.to_string(),
                arm: i % 2,
                probabilities: vec![0.5; 2],
                outcome: if i % 2 == 0 { 2.0 } else { 5.0 },
                population_weight: 1.0,
            })
            .collect::<Vec<_>>();
        let prior = RandomizedGaussianPrior {
            mean: vec![0.0, 1.0],
            variance: vec![4.0, 9.0],
            outcome_variance: vec![2.0; 20],
            draws: 20000,
            seed: 17,
        };
        let posterior = randomized_joint_posterior(&design, &units, &prior).unwrap();
        for j in 0..2 {
            let reference_variance = 1.0 / (1.0 / prior.variance[j] + 5.0);
            let reference_mean = reference_variance
                * (prior.mean[j] / prior.variance[j] + if j == 0 { 10.0 } else { 25.0 });
            assert!(
                (posterior.estimate.means[j] - reference_mean).abs()
                    < 6.0 * (reference_variance / prior.draws as f64).sqrt()
            );
            assert!((posterior.estimate.covariance[j * 2 + j] - reference_variance).abs() < 0.015);
        }
        assert!(posterior.estimate.covariance[1].abs() < 0.01);
    }
}

/// Proper finite-population potential-response model for unequal cluster populations.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RandomizedFinitePopulationPrior {
    /// Proper common arm-mean prior and known observation variances.
    pub arm: RandomizedGaussianPrior,
    /// Known positive between-unit latent potential-mean variances, one per arm.
    /// Latent unit-by-arm means are conditionally independent given the arm mean.
    pub heterogeneity_variance: Vec<f64>,
}
/// Posterior of the actual fixed population-weighted mean of unit potential responses.
/// Unit-level latent response uncertainty is retained for both observed and unobserved
/// assignments. Treating the common arm mean alone as a finite-population weighted
/// average would incorrectly remove this uncertainty. All weights are preassignment.
/// # Errors
/// Invalid randomization, proper prior/variance or finite-population weights.
pub fn randomized_weighted_population_posterior(
    design: &JointRandomizedDesign,
    units: &[JointRandomizedUnit],
    prior: &RandomizedFinitePopulationPrior,
) -> Result<super::JointDesignPosterior, String> {
    let _identified = randomized_joint_means(design, units)?;
    let k = design.worlds.len();
    let arm = &prior.arm;
    if arm.mean.len() != k
        || arm.variance.len() != k
        || arm.outcome_variance.len() != units.len()
        || prior.heterogeneity_variance.len() != k
        || arm.draws < 2
        || arm.draws > 100_000
        || arm.mean.iter().any(|v| !v.is_finite())
        || arm
            .variance
            .iter()
            .chain(&arm.outcome_variance)
            .chain(&prior.heterogeneity_variance)
            .any(|v| !v.is_finite() || *v <= 0.0)
    {
        return Err("weighted finite-population posterior requires proper Gaussian arm/unit priors and known positive observation/heterogeneity variances".into());
    }
    let mut precision = arm.variance.iter().map(|v| 1.0 / v).collect::<Vec<_>>();
    let mut weighted = arm.mean.iter().zip(&precision).map(|(m, p)| m * p).collect::<Vec<_>>();
    for (i, u) in units.iter().enumerate() {
        let variance = prior.heterogeneity_variance[u.arm] + arm.outcome_variance[i];
        precision[u.arm] += 1.0 / variance;
        weighted[u.arm] += u.outcome / variance;
    }
    let total = units.iter().map(|u| u.population_weight).sum::<f64>();
    let mut rng = antecedent_core::CausalRng::from_seed(arm.seed);
    let mut draws = Vec::with_capacity(arm.draws);
    for _ in 0..arm.draws {
        let mut means = Vec::with_capacity(k);
        for j in 0..k {
            let theta = weighted[j] / precision[j]
                + antecedent_kernels::standard_normal(&mut rng) / precision[j].sqrt();
            let tau = prior.heterogeneity_variance[j];
            let mut value = 0.0;
            for (i, u) in units.iter().enumerate() {
                let (conditional_mean, conditional_variance) = if u.arm == j {
                    let sigma = arm.outcome_variance[i];
                    ((sigma * theta + tau * u.outcome) / (tau + sigma), tau * sigma / (tau + sigma))
                } else {
                    (theta, tau)
                };
                let latent = conditional_mean
                    + conditional_variance.sqrt() * antecedent_kernels::standard_normal(&mut rng);
                value += u.population_weight * latent / total;
            }
            means.push(value);
        }
        draws.push(means);
    }
    super::JointDesignPosterior::from_draws(design.worlds.clone(),draws,design.sampling_unit.clone(),"fixed population-weighted latent unit potential-response means; Gaussian outcome measurement likelihood with declared known variance; conditionally independent Gaussian unit-by-arm potential-mean priors around proper Gaussian arm means with declared known heterogeneity variances; all original population weights retained".into())
}

#[cfg(test)]
mod weighted_posterior_tests {
    use super::*;
    #[test]
    fn unequal_geo_populations_retain_unobserved_potential_response_uncertainty() {
        let design = JointRandomizedDesign {
            worlds: vec!["base".into(), "joint".into()],
            sampling_unit: "geo".into(),
            population: "fixed weighted census".into(),
            consistency: "whole-program geo assignment".into(),
            between_unit_isolation: "geo clusters isolate interference".into(),
        };
        let units = (0..8)
            .map(|i| JointRandomizedUnit {
                id: i.to_string(),
                arm: i % 2,
                probabilities: vec![0.5; 2],
                outcome: if i % 2 == 0 { 2.0 } else { 5.0 },
                population_weight: (i + 1) as f64,
            })
            .collect::<Vec<_>>();
        let prior = RandomizedFinitePopulationPrior {
            arm: RandomizedGaussianPrior {
                mean: vec![0.0, 0.0],
                variance: vec![4.0; 2],
                outcome_variance: vec![1.0; 8],
                draws: 20000,
                seed: 19,
            },
            heterogeneity_variance: vec![2.0; 2],
        };
        let posterior = randomized_weighted_population_posterior(&design, &units, &prior).unwrap();
        let total = units.iter().map(|u| u.population_weight).sum::<f64>();
        for j in 0..2 {
            let theta_var = 1.0 / (0.25 + 4.0 / 3.0);
            let theta_mean = theta_var * (if j == 0 { 8.0 } else { 20.0 }) / 3.0;
            let mut slope = 0.0;
            let mut offset = 0.0;
            let mut conditional_var = 0.0;
            for u in &units {
                let w = u.population_weight / total;
                if u.arm == j {
                    slope += w / 3.0;
                    offset += w * 2.0 * u.outcome / 3.0;
                    conditional_var += w * w * 2.0 / 3.0;
                } else {
                    slope += w;
                    conditional_var += w * w * 2.0;
                }
            }
            let reference_mean = slope * theta_mean + offset;
            let reference_var = slope * slope * theta_var + conditional_var;
            assert!(
                (posterior.estimate.means[j] - reference_mean).abs()
                    < 6.0 * (reference_var / prior.arm.draws as f64).sqrt()
            );
            assert!((posterior.estimate.covariance[j * 2 + j] - reference_var).abs() < 0.025);
            assert!(reference_var > theta_var * 0.25);
        }
    }
}
