//! Joint synthetic-rollout targets from explicit unaffected donor/reference worlds.
// SPDX-License-Identifier: MIT OR Apache-2.0
use super::JointDesignPosterior;
use crate::synthetic_rollout::{SyntheticRolloutPanel, conformal_synthetic_control_at_level};
use antecedent_core::ExecutionContext;
use std::{collections::BTreeSet, sync::Arc};
/// One observed unit-time row in the original rollout panel.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JointSyntheticObservation {
    /// Original unit identity.
    pub unit: String,
    /// Original calendar period.
    pub time: i64,
    /// Observed outcome.
    pub outcome: f64,
}
/// A complete rollout target and its admitted, explicitly unaffected donor units.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JointSyntheticWorld {
    /// Stable rollout target key.
    pub key: String,
    /// Unit exposed to the complete joint intervention program.
    pub treated_unit: String,
    /// First intervention period.
    pub intervention: i64,
    /// Explicit unaffected donors; other treated programs are never silently pooled.
    pub donors: Vec<String>,
}
/// Scientific identifying restriction of a joint synthetic counterfactual model.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JointSyntheticDesign {
    /// Supported observed joint rollout contrasts and donor-reference worlds.
    pub worlds: Vec<JointSyntheticWorld>,
    /// Convex-hull untreated counterfactual model and its pre/post persistence.
    pub counterfactual_model: String,
    /// Donor isolation, no anticipation and no contamination by joint programs.
    pub donor_isolation: String,
    /// Unit/cohort populations and the time averaging target.
    pub population: String,
}
/// Retained donor fit and simultaneous uncertainty for one joint rollout target.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct JointSyntheticCoordinate {
    /// Native target key.
    pub key: String,
    /// Average post-period effect on this treated unit/cohort.
    pub effect: f64,
    /// Donor weights retained in original unit identities.
    pub donor_weights: Vec<(String, f64)>,
    /// Preperiod root mean squared counterfactual fit error.
    pub preperiod_rmse: f64,
    /// Constant-effect conformal interval, None means unbounded rather than zero uncertainty.
    pub interval: Option<[f64; 2]>,
}
/// Joint synthetic-control inference with explicit family-wise conformal level.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct JointSyntheticConformal {
    /// Response coordinates and full donor evidence.
    pub coordinates: Vec<JointSyntheticCoordinate>,
    /// Joint Bonferroni level, valid only under retained exchangeable-residual model.
    pub joint_level: f64,
    /// Additional inferential premise, separate from causal counterfactual identification.
    pub residual_exchangeability: String,
}
impl JointSyntheticConformal {
    /// Project the simultaneous region onto a prespecified joint-program contrast.
    /// # Errors
    /// Missing/invalid coefficients or any unbounded component.
    pub fn contrast_interval(&self, coefficients: &[f64]) -> Result<(f64, [f64; 2]), String> {
        if coefficients.len() != self.coordinates.len()
            || coefficients.iter().any(|v| !v.is_finite())
        {
            return Err("invalid joint synthetic contrast coefficients".into());
        }
        let mut mean = 0.0;
        let mut lower = 0.0;
        let mut upper = 0.0;
        for (c, x) in coefficients.iter().zip(&self.coordinates) {
            mean += c * x.effect;
            if *c == 0.0 {
                continue;
            }
            let [a, b] =
                x.interval.ok_or("joint synthetic contrast has an unbounded response component")?;
            lower += if *c > 0.0 { c * a } else { c * b };
            upper += if *c > 0.0 { c * b } else { c * a };
        }
        Ok((mean, [lower, upper]))
    }
}
fn validate(
    design: &JointSyntheticDesign,
    data: &[JointSyntheticObservation],
) -> Result<(), String> {
    if design.worlds.is_empty()
        || design.worlds.len() > 64
        || data.is_empty()
        || data.len() > 1_000_000
        || [&design.counterfactual_model, &design.donor_isolation, &design.population]
            .iter()
            .any(|s| s.trim().is_empty())
        || design.worlds.iter().map(|w| &w.key).collect::<BTreeSet<_>>().len()
            != design.worlds.len()
    {
        return Err("joint synthetic design requires explicit counterfactual/donor/population premises and distinct targets".into());
    }
    for world in &design.worlds {
        if world.key.is_empty()
            || world.treated_unit.is_empty()
            || world.intervention <= 0
            || world.donors.len() < 3
            || world.donors.len() > 1024
            || world.donors.iter().collect::<BTreeSet<_>>().len() != world.donors.len()
            || world.donors.iter().any(|d| {
                d.is_empty()
                    || d == &world.treated_unit
                    || design.worlds.iter().any(|w| &w.treated_unit == d)
            })
        {
            return Err("joint synthetic donor set must explicitly exclude every treated program and contain sufficient unaffected reference units".into());
        }
    }
    let mut rows = BTreeSet::new();
    for row in data {
        if row.unit.is_empty()
            || row.time <= 0
            || !row.outcome.is_finite()
            || !rows.insert((&row.unit, row.time))
        {
            return Err("invalid or duplicated original synthetic unit-time row".into());
        }
    }
    Ok(())
}
fn subset(
    world: &JointSyntheticWorld,
    data: &[JointSyntheticObservation],
) -> (Vec<f64>, Vec<String>, Vec<i64>) {
    let rows = data
        .iter()
        .filter(|row| row.unit == world.treated_unit || world.donors.contains(&row.unit))
        .collect::<Vec<_>>();
    (
        rows.iter().map(|r| r.outcome).collect(),
        rows.iter().map(|r| r.unit.clone()).collect(),
        rows.iter().map(|r| r.time).collect(),
    )
}
/// Fit every requested joint rollout against its supported unaffected donor world.
/// Bonferroni inversion gives one simultaneous region without assuming independent
/// marginal effects. Native weights, preperiod fit and counterfactual restrictions
/// survive projection. The reported release calibration status remains unmeasured.
/// # Errors
/// Missing design or exchangeability premises, contaminated/unsupported donor worlds,
/// insufficient conformal resolution, cancellation or admission refusal.
pub fn synthetic_joint_conformal(
    design: &JointSyntheticDesign,
    data: &[JointSyntheticObservation],
    joint_level: f64,
    residual_exchangeability: &str,
    ctx: &ExecutionContext,
) -> Result<JointSyntheticConformal, String> {
    validate(design, data)?;
    if residual_exchangeability.trim().is_empty()
        || !joint_level.is_finite()
        || joint_level <= 0.0
        || joint_level >= 1.0
    {
        return Err("joint synthetic conformal inference needs a finite family-wise level and exchangeable counterfactual residual premise".into());
    }
    let marginal = 1.0 - (1.0 - joint_level) / design.worlds.len() as f64;
    let mut coordinates = Vec::new();
    for world in &design.worlds {
        let (outcome, labels, periods) = subset(world, data);
        let units = labels.iter().map(|x| Arc::<str>::from(x.as_str())).collect::<Vec<_>>();
        let fit = conformal_synthetic_control_at_level(
            &SyntheticRolloutPanel { outcome: &outcome, units: &units, periods: &periods },
            &world.treated_unit,
            world.intervention,
            marginal,
            ctx,
        )
        .map_err(|e| e.to_string())?;
        coordinates.push(JointSyntheticCoordinate {
            key: world.key.clone(),
            effect: fit.overall.estimate,
            donor_weights: fit.donor_weights,
            preperiod_rmse: fit.pre_treatment_rmse,
            interval: fit.overall.interval,
        });
    }
    Ok(JointSyntheticConformal {
        coordinates,
        joint_level,
        residual_exchangeability: residual_exchangeability.into(),
    })
}
/// Statistical assumptions and proper priors for conditional Gaussian-AR1 donor models.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JointSyntheticPrior {
    /// Symmetric proper simplex donor-weight prior concentration.
    pub dirichlet_concentration: f64,
    /// Inverse-gamma innovation variance shape.
    pub variance_shape: f64,
    /// Inverse-gamma innovation variance scale.
    pub variance_scale: f64,
    /// Known AR1 coefficient, absolute value below one.
    pub ar1_rho: f64,
    /// Burn-in sweeps per chain.
    pub burn_in: usize,
    /// Total posterior draws over four chains.
    pub draws: usize,
    /// Deterministic stream seed.
    pub seed: u64,
    /// Required conditional independence of distinct treated residual processes,
    /// given observed donors. Shared donors are fixed predictors, not independent evidence.
    pub independent_treated_residuals: String,
}
/// Joint posterior from actual simplex donor-model likelihood updates, using the
/// existing convergence-gated Bayesian synthetic-control owner. Distinct treated
/// response factors use independent streams and priors conditional on the common
/// observed donor paths; shared donor data are never counted as extra outcomes.
/// # Errors
/// Invalid scientific or residual-independence premises, unsupported donor worlds,
/// invalid proper priors or failed sampler convergence/ESS checks.
pub fn synthetic_joint_posterior(
    design: &JointSyntheticDesign,
    data: &[JointSyntheticObservation],
    prior: &JointSyntheticPrior,
) -> Result<JointDesignPosterior, String> {
    validate(design, data)?;
    if prior.independent_treated_residuals.trim().is_empty()
        || design.worlds.iter().map(|w| &w.treated_unit).collect::<BTreeSet<_>>().len()
            != design.worlds.len()
        || prior.draws > 100_000
        || prior.draws < 4
    {
        return Err("joint Bayesian synthetic inference requires distinct treated-unit likelihood factors and explicit conditional residual independence".into());
    }
    let mut draws = vec![vec![0.0; design.worlds.len()]; prior.draws - prior.draws % 4];
    for (index, world) in design.worlds.iter().enumerate() {
        let (outcome, units, periods) = subset(world, data);
        let fit = crate::bayesian_synthetic_control::fit_bayesian_synthetic_control(
            &outcome,
            &units,
            &periods,
            &crate::bayesian_synthetic_control::BayesianSyntheticControlSpec {
                treated_unit: &world.treated_unit,
                intervention_period: world.intervention,
                dirichlet_concentration: prior.dirichlet_concentration,
                variance_shape: prior.variance_shape,
                variance_scale: prior.variance_scale,
                ar1_rho: prior.ar1_rho,
                burn_in: prior.burn_in,
            },
            crate::util::DrawSpec {
                draws: prior.draws,
                seed: prior.seed.wrapping_add(0x9e37_79b9_7f4a_7c15_u64.wrapping_mul(index as u64 + 1)),
            },
        )
        .map_err(|e| e.to_string())?;
        for (row, value) in draws.iter_mut().zip(fit.effect_draws) {
            row[index] = value;
        }
    }
    JointDesignPosterior::from_draws(design.worlds.iter().map(|w|w.key.clone()).collect(),draws,"treated-unit counterfactual Gaussian-AR1 model conditional on observed donor paths".into(),"native convergence-gated Gaussian AR1 synthetic-control likelihood with proper Dirichlet simplex donor priors and inverse-gamma innovation-variance priors; independent distinct treated residual factors conditional on common observed unaffected donors".into())
}

/// Execute the donor-supported joint rollout point functional with full fit evidence.
/// No interval or covariance is manufactured when a statistical procedure is absent.
/// # Errors
/// Missing model/donor premises, malformed panel or insufficient donor/preperiod support.
pub fn synthetic_joint_point(
    design: &JointSyntheticDesign,
    data: &[JointSyntheticObservation],
) -> Result<Vec<JointSyntheticCoordinate>, String> {
    validate(design, data)?;
    let mut coordinates = Vec::with_capacity(design.worlds.len());
    for world in &design.worlds {
        let (outcome, units, periods) = subset(world, data);
        let fit = crate::synthetic_control::fit_synthetic_control(
            &outcome,
            &units,
            &periods,
            &world.treated_unit,
            world.intervention,
        )?;
        coordinates.push(JointSyntheticCoordinate {
            key: world.key.clone(),
            effect: fit.effect,
            donor_weights: fit.donor_weights,
            preperiod_rmse: fit.pre_treatment_rmse,
            interval: None,
        });
    }
    Ok(coordinates)
}
