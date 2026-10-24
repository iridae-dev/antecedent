//! Joint rollout difference-in-differences using supported untreated comparison cohorts.
// SPDX-License-Identifier: MIT OR Apache-2.0
use super::{JointDesignEstimate, JointUncertainty};
use std::collections::{BTreeMap, BTreeSet};
/// An independent panel unit assigned one complete simultaneous rollout regime.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JointRolloutUnit {
    /// Original independent unit identity.
    pub id: String,
    /// Complete multichannel regime, not a marginal treatment indicator.
    pub regime: String,
    /// First adoption time; `None` denotes never treated.
    pub adoption: Option<i64>,
    /// Observed time and outcome pairs.
    pub outcomes: Vec<(i64, f64)>,
}
/// An explicit cohort-time contrast within a joint rollout regime.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JointRolloutCell {
    /// Stable world/contrast key.
    pub key: String,
    /// Complete regime whose adopting cohort is targeted.
    pub regime: String,
    /// Cohort adoption time.
    pub adoption: i64,
    /// Post-adoption target time.
    pub time: i64,
    /// Pre-adoption baseline, chosen before anticipation begins.
    pub baseline: i64,
}
/// Retained scientific premises of cohort-time joint rollout contrasts.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JointRolloutDesign {
    /// Every requested cohort-time target, which may be assembled into program interactions.
    pub cells: Vec<JointRolloutCell>,
    /// Explicit no-anticipation restriction.
    pub no_anticipation: String,
    /// Parallel untreated trends for each targeted cohort and its retained comparisons.
    pub parallel_trends: String,
    /// Consistency and no between-unit interference restriction.
    pub consistency_and_isolation: String,
    /// Population is the adopting cohort, never silently relabelled a population ATE.
    pub population: String,
}
fn point(design: &JointRolloutDesign, units: &[&JointRolloutUnit]) -> Result<Vec<f64>, String> {
    design.cells.iter().map(|cell|{
        let change=|u:&&JointRolloutUnit|{
            let b=u.outcomes.iter().find(|(t,_)|*t==cell.baseline)?.1;
            let y=u.outcomes.iter().find(|(t,_)|*t==cell.time)?.1;Some(y-b)
        };
        let treated=units.iter().filter(|u|u.regime==cell.regime&&u.adoption==Some(cell.adoption)).filter_map(change).collect::<Vec<_>>();
        // A cohort treated in a different regime is not a valid untreated donor.
        let controls=units.iter().filter(|u|u.adoption.is_none_or(|g|g>cell.time.max(cell.baseline))).filter_map(change).collect::<Vec<_>>();
        if treated.is_empty()||controls.is_empty(){return Err(format!("joint rollout {} lacks treated/comparison cohort support at both retained periods",cell.key));}
        Ok(treated.iter().sum::<f64>()/treated.len() as f64-controls.iter().sum::<f64>()/controls.len() as f64)
    }).collect()
}
/// Estimate joint cohort-time ATTs against never/not-yet-treated comparisons.
/// Already treated units are excluded from controls. Whole-unit delete-one
/// realizations refit every cell and retain dependence among regime contrasts.
///
/// # Errors
/// Missing parallel-trend/anticipation premises, malformed panels, unsupported
/// cohorts or uncertainty realizations losing required cohort support.
pub fn rollout_joint_effects(
    design: &JointRolloutDesign,
    units: &[JointRolloutUnit],
) -> Result<JointDesignEstimate, String> {
    if design.cells.is_empty()
        || design.cells.len() > 256
        || units.len() > 16384
        || units.len() < 4
        || [
            &design.no_anticipation,
            &design.parallel_trends,
            &design.consistency_and_isolation,
            &design.population,
        ]
        .iter()
        .any(|s| s.trim().is_empty())
        || design.cells.iter().any(|c| {
            c.key.is_empty()
                || c.regime.is_empty()
                || c.baseline >= c.adoption
                || c.time < c.adoption
        })
        || design.cells.iter().map(|c| &c.key).collect::<BTreeSet<_>>().len() != design.cells.len()
    {
        return Err("joint rollout requires distinct valid cohort-time targets and explicit scientific premises".into());
    }
    let mut ids = BTreeSet::new();
    for u in units {
        let panel = u.outcomes.iter().copied().collect::<BTreeMap<_, _>>();
        if u.id.is_empty()
            || !ids.insert(&u.id)
            || panel.len() != u.outcomes.len()
            || u.outcomes.iter().any(|(_, y)| !y.is_finite())
        {
            return Err("invalid joint rollout independent-unit panel".into());
        }
    }
    let refs = units.iter().collect::<Vec<_>>();
    let means = point(design, &refs)?;
    let n = units.len();
    let k = means.len();
    let mut deletes = Vec::with_capacity(n);
    for omitted in 0..n {
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
        worlds: design.cells.iter().map(|c| c.key.clone()).collect(),
        means,
        covariance,
        sampling_unit: "independent rollout unit".into(),
        uncertainty: JointUncertainty::DeleteEpisodeJackknife,
    })
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn regime_interactions_use_only_supported_untreated_comparisons() {
        let mut units = Vec::new();
        for (regime, effect) in [("00", 0.0), ("10", 2.0), ("01", 3.0), ("11", 9.0)] {
            for i in 0..3 {
                units.push(JointRolloutUnit {
                    id: format!("{regime}/{i}"),
                    regime: regime.into(),
                    adoption: (regime != "00").then_some(2),
                    outcomes: vec![(1, i as f64), (2, i as f64 + 5.0 + effect)],
                });
            }
        }
        let design = JointRolloutDesign {
            cells: ["10", "01", "11"]
                .into_iter()
                .map(|regime| JointRolloutCell {
                    key: regime.into(),
                    regime: regime.into(),
                    adoption: 2,
                    time: 2,
                    baseline: 1,
                })
                .collect(),
            no_anticipation: "zero effects before adoption".into(),
            parallel_trends: "untreated increments common to all cohorts".into(),
            consistency_and_isolation: "joint versions fixed and independent units".into(),
            population: "cohort 2".into(),
        };
        let fit = rollout_joint_effects(&design, &units).unwrap();
        assert_eq!(fit.means, vec![2.0, 3.0, 9.0]);
        assert_eq!(fit.contrast(&[-1.0, -1.0, 1.0]).unwrap().0, 4.0);
        let mut invalid = design.clone();
        invalid.parallel_trends.clear();
        assert!(rollout_joint_effects(&invalid, &units).is_err());
        units.retain(|u| u.adoption.is_some());
        assert!(rollout_joint_effects(&design, &units).is_err());
    }
}

/// Proper joint panel likelihood and priors for regime/cohort/time effects.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JointRolloutGaussianPrior {
    /// Proper common prior mean of unit intercepts.
    pub unit_mean: f64,
    /// Positive unit-intercept prior variance.
    pub unit_variance: f64,
    /// Proper prior mean of calendar effects (earliest calendar effect fixed to zero).
    pub time_mean: f64,
    /// Positive calendar-effect prior variance.
    pub time_variance: f64,
    /// Proper prior means for every requested regime/cohort/time ATT.
    pub effect_mean: Vec<f64>,
    /// Positive effect prior variances in target order.
    pub effect_variance: Vec<f64>,
    /// Known marginal Gaussian observation residual variance.
    pub outcome_variance: f64,
    /// Known within-unit calendar-time AR1 residual coefficient.
    pub ar1_rho: f64,
    /// Aligned posterior mechanism draws.
    pub draws: usize,
    /// Deterministic stream seed.
    pub seed: u64,
}
/// Joint Bayesian regime/cohort/time panel likelihood, retaining cohort ATT targets.
/// Unit effects, calendar effects and every observed post-adoption cohort-time
/// effect are fitted together with proper Gaussian priors and declared known
/// AR1 outcome errors. Already-treated units have their own saturated effects;
/// their unmodeled treatment responses are never used as untreated comparisons.
/// Native untreated-comparison support and causal premises are checked first.
/// # Errors
/// Invalid design/support, omitted observed treated cohort-time effect, invalid proper
/// prior/covariance, excessive dimensions or numerical posterior failure.
pub fn rollout_joint_posterior(
    design: &JointRolloutDesign,
    units: &[JointRolloutUnit],
    prior: &JointRolloutGaussianPrior,
) -> Result<super::JointDesignPosterior, String> {
    let _identified = rollout_joint_effects(design, units)?;
    let times = units
        .iter()
        .flat_map(|u| u.outcomes.iter().map(|(t, _)| *t))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let k = design.cells.len();
    let n = units.len();
    let parameters = n + times.len() - 1 + k;
    if parameters > 2048
        || prior.effect_mean.len() != k
        || prior.effect_variance.len() != k
        || !prior.unit_mean.is_finite()
        || !prior.time_mean.is_finite()
        || prior.effect_mean.iter().any(|x| !x.is_finite())
        || [prior.unit_variance, prior.time_variance, prior.outcome_variance]
            .iter()
            .chain(&prior.effect_variance)
            .any(|v| !v.is_finite() || *v <= 0.0)
        || !prior.ar1_rho.is_finite()
        || prior.ar1_rho.abs() >= 1.0
        || prior.draws < 2
        || prior.draws > 100000
    {
        return Err("joint rollout posterior needs proper priors, known nonsingular residual covariance and bounded parameter dimensions".into());
    }
    for u in units {
        if let Some(g) = u.adoption {
            for &(time, _) in &u.outcomes {
                if time >= g
                    && !design
                        .cells
                        .iter()
                        .any(|c| c.regime == u.regime && c.adoption == g && c.time == time)
                {
                    return Err("joint rollout likelihood must explicitly model every observed treated regime/cohort/time cell; omitted treatment effects cannot become untreated controls".into());
                }
            }
        }
    }
    let offset = n + times.len() - 1;
    let mut precision = vec![0.0; parameters * parameters];
    let mut rhs = vec![0.0; parameters];
    for j in 0..parameters {
        let (mean, variance) = if j < n {
            (prior.unit_mean, prior.unit_variance)
        } else if j < offset {
            (prior.time_mean, prior.time_variance)
        } else {
            (prior.effect_mean[j - offset], prior.effect_variance[j - offset])
        };
        precision[j * parameters + j] = 1.0 / variance;
        rhs[j] = mean / variance;
    }
    for (index, u) in units.iter().enumerate() {
        let sorted = u.outcomes.iter().copied().collect::<BTreeMap<_, _>>();
        let mut previous: Option<(i64, f64, Vec<f64>)> = None;
        for (&time, &outcome) in &sorted {
            let mut row = vec![0.0; parameters];
            row[index] = 1.0;
            let calendar =
                times.binary_search(&time).map_err(|_| "rollout calendar axis changed")?;
            if calendar > 0 {
                row[n + calendar - 1] = 1.0;
            }
            if let Some(g) = u.adoption {
                if time >= g {
                    let cell = design
                        .cells
                        .iter()
                        .position(|c| c.regime == u.regime && c.adoption == g && c.time == time)
                        .ok_or("unmodeled cohort-time effect")?;
                    row[offset + cell] = 1.0;
                }
            }
            let (response, variance, whitened) =
                if let Some((before, before_y, before_row)) = &previous {
                    let gap = time
                        .checked_sub(*before)
                        .and_then(|t| i32::try_from(t).ok())
                        .ok_or("AR1 calendar gap is outside representable range")?;
                    let rho = prior.ar1_rho.powi(gap);
                    (
                        outcome - rho * before_y,
                        prior.outcome_variance * (1.0 - rho * rho),
                        row.iter().zip(before_row).map(|(x, b)| x - rho * b).collect::<Vec<_>>(),
                    )
                } else {
                    (outcome, prior.outcome_variance, row.clone())
                };
            if variance <= 0.0 || !variance.is_finite() {
                return Err("rollout AR1 innovation variance degenerate".into());
            }
            let nonzero =
                whitened.iter().enumerate().filter(|(_, x)| **x != 0.0).collect::<Vec<_>>();
            for &(j, x) in &nonzero {
                rhs[j] += x * response / variance;
                for &(l, y) in &nonzero {
                    precision[j * parameters + l] += x * y / variance;
                }
            }
            previous = Some((time, outcome, row));
        }
    }
    let chol = antecedent_stats::cholesky_spd(&precision, parameters)
        .ok_or("rollout posterior precision not positive definite")?;
    let mean = antecedent_stats::chol_solve(&chol, parameters, &rhs)
        .ok_or("rollout posterior solve failed")?;
    let mut rng = antecedent_core::CausalRng::from_seed(prior.seed);
    let mut draws = Vec::with_capacity(prior.draws);
    for _ in 0..prior.draws {
        let mut delta = (0..parameters)
            .map(|_| antecedent_kernels::standard_normal(&mut rng))
            .collect::<Vec<_>>();
        for j in (0..parameters).rev() {
            let sum = (j + 1..parameters).map(|l| chol[l * parameters + j] * delta[l]).sum::<f64>();
            delta[j] = (delta[j] - sum) / chol[j * parameters + j];
        }
        draws.push((0..k).map(|j| mean[offset + j] + delta[offset + j]).collect());
    }
    super::JointDesignPosterior::from_draws(design.cells.iter().map(|c|c.key.clone()).collect(),draws,"independent complete rollout panel unit".into(),"Gaussian joint unit/calendar/saturated regime-cohort-time likelihood with declared known within-unit AR1 covariance and proper Gaussian priors on all unit, time and target effects; native no-anticipation and untreated-comparison parallel-trend target retained".into())
}
