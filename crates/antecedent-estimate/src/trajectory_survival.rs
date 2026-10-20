//! Survival under a prescribed time-varying treatment, on subject trajectories.
//!
//! Each subject contributes one discrete-time trajectory. A trajectory is a sequence of
//! treatments, not a static trial arm. At period `k` the cumulative weight of a subject who
//! has followed the regime and is still at risk is
//!
//! ```text
//! ∏_{j=0}^{k} censoring_weight_j / P(A_j = a_j | history_j).
//! ```
//!
//! The censoring factor is the caller-supplied inverse probability of still being observed.
//! It must be finite and positive. Treatment probabilities and censoring weights are inputs.
//! This module does not fit them.
//!
//! [`trajectory_survival_contrast`] is the weighted product-limit contrast of two regimes.
//! The risk set at each period is the subjects who followed that regime so far. No interval
//! is issued.
//!
//! [`trajectory_hazard_posterior`] is a different object. Conditional on the same weights,
//! each period hazard has an independent beta posterior from the weighted Bernoulli
//! likelihood of that risk set. The reported survival is the posterior mean of the product
//! of period survivals. It is not the product-limit, and it does not model the treatment or
//! censoring mechanisms that produced the weights.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

/// Subject-major discrete-time trajectories.
///
/// Every slice has length `subjects * periods`, stored subject-major. `at_risk` means the
/// subject is still observed and event-free at that period, so the subject can enter the
/// risk set. Leaving the risk set is permanent. An event is recorded only while at risk
/// and removes the subject afterwards.
#[derive(Clone, Copy, Debug)]
pub struct TrajectoryRows<'a> {
    /// Number of independent subject trajectories.
    pub subjects: usize,
    /// Decision periods on each trajectory.
    pub periods: usize,
    /// Observed treatment at each subject-period.
    pub treatment: &'a [bool],
    /// Event indicator at each subject-period.
    pub event: &'a [bool],
    /// Whether the subject is still observed and event-free.
    pub at_risk: &'a [bool],
    /// Conditional probability of the observed treatment.
    pub treatment_probability: &'a [f64],
    /// Positive period-specific censoring weight.
    pub censoring_weight: &'a [f64],
    /// Positivity floor for a treatment probability, in `(0, 0.5]`.
    pub minimum_probability: f64,
}

/// Weighted product-limit survival for one treatment trajectory.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrajectorySurvival {
    /// Survival through the last period.
    pub survival: f64,
    /// Restricted mean over unit-length periods, using survival at each period's start.
    pub restricted_mean: f64,
    /// Smallest positive weighted risk set along the trajectory.
    pub minimum_risk_weight: f64,
    /// Largest cumulative weight in any risk set.
    pub maximum_weight: f64,
}

/// Active-minus-control weighted product-limit contrast.
///
/// Both curves run through every period. This is a point. It is not an interval.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrajectorySurvivalContrast {
    /// Weighted product-limit for the active trajectory.
    pub active: TrajectorySurvival,
    /// Weighted product-limit for the control trajectory.
    pub control: TrajectorySurvival,
    /// Active survival minus control survival at the horizon.
    pub survival_difference: f64,
    /// Active restricted mean minus control restricted mean.
    pub restricted_mean_difference: f64,
}

/// Positive counts of an independent beta prior on a period hazard.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HazardPrior {
    /// Prior event count. Finite and positive.
    pub events: f64,
    /// Prior non-event count. Finite and positive.
    pub nonevents: f64,
}

impl HazardPrior {
    /// Uniform `Beta(1, 1)` prior.
    #[must_use]
    pub const fn uniform() -> Self {
        Self { events: 1.0, nonevents: 1.0 }
    }
}

/// Posterior means of survival under two trajectories.
///
/// The period hazards are independent after conditioning on the supplied weights. These
/// means are not a weighted product-limit and not a calibrated credible interval.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrajectoryHazardPosterior {
    /// Posterior mean of active survival through the horizon.
    pub active_survival_mean: f64,
    /// Posterior mean of control survival through the horizon.
    pub control_survival_mean: f64,
    /// Active mean minus control mean.
    pub survival_difference_mean: f64,
    /// Posterior mean of the active restricted mean.
    pub active_restricted_mean: f64,
    /// Posterior mean of the control restricted mean.
    pub control_restricted_mean: f64,
    /// Active restricted-mean posterior mean minus the control mean.
    pub restricted_mean_difference_mean: f64,
}

/// Weighted product-limit contrast of two treatment trajectories.
///
/// # Errors
///
/// Returns an error when the rows are misaligned, a censoring weight is not finite and
/// positive, a treatment probability leaves the declared floor, a subject re-enters after
/// leaving the risk set, an event is coded off the risk set, the trajectories are not two
/// distinct period sequences, a cumulative weight is not finite and positive, or either
/// trajectory has an empty weighted risk set at any period.
#[must_use = "the contrast is the weighted product-limit of the two trajectories"]
pub fn trajectory_survival_contrast(
    rows: &TrajectoryRows<'_>,
    active: &[bool],
    control: &[bool],
) -> Result<TrajectorySurvivalContrast, &'static str> {
    let (active_mass, control_mass) = regime_masses(rows, active, control)?;
    let active = product_limit(&active_mass);
    let control = product_limit(&control_mass);
    Ok(TrajectorySurvivalContrast {
        survival_difference: active.survival - control.survival,
        restricted_mean_difference: active.restricted_mean - control.restricted_mean,
        active,
        control,
    })
}

/// Independent weighted-hazard posterior for the same two trajectories.
///
/// The beta update at each period adds the cumulative weights of followers who remain at
/// risk. Prior counts stay positive, so the posterior mean does not collapse to the
/// product-limit.
///
/// # Errors
///
/// Returns every error of [`trajectory_survival_contrast`], and an error when a prior
/// count is not finite and positive.
#[must_use = "the posterior mean is not the product-limit"]
pub fn trajectory_hazard_posterior(
    rows: &TrajectoryRows<'_>,
    active: &[bool],
    control: &[bool],
    prior: HazardPrior,
) -> Result<TrajectoryHazardPosterior, &'static str> {
    if !prior.events.is_finite()
        || !prior.nonevents.is_finite()
        || prior.events <= 0.0
        || prior.nonevents <= 0.0
    {
        return Err("hazard prior counts must be finite and positive");
    }
    let (active_mass, control_mass) = regime_masses(rows, active, control)?;
    let (active_survival_mean, active_restricted_mean) = posterior_means(&active_mass, prior);
    let (control_survival_mean, control_restricted_mean) = posterior_means(&control_mass, prior);
    Ok(TrajectoryHazardPosterior {
        active_survival_mean,
        control_survival_mean,
        survival_difference_mean: active_survival_mean - control_survival_mean,
        active_restricted_mean,
        control_restricted_mean,
        restricted_mean_difference_mean: active_restricted_mean - control_restricted_mean,
    })
}

struct PeriodMass {
    risk: f64,
    event: f64,
    nonevent: f64,
    maximum_weight: f64,
}

fn regime_masses(
    rows: &TrajectoryRows<'_>,
    active: &[bool],
    control: &[bool],
) -> Result<(Vec<PeriodMass>, Vec<PeriodMass>), &'static str> {
    validate_rows(rows)?;
    if active.len() != rows.periods || control.len() != rows.periods || active == control {
        return Err("treatment trajectories must have one action per period and must differ");
    }
    Ok((period_masses(rows, active)?, period_masses(rows, control)?))
}

fn validate_rows(rows: &TrajectoryRows<'_>) -> Result<(), &'static str> {
    let TrajectoryRows {
        subjects,
        periods,
        treatment,
        event,
        at_risk,
        treatment_probability,
        censoring_weight,
        minimum_probability,
    } = *rows;
    let cells = subjects.checked_mul(periods).ok_or("trajectory dimensions overflow")?;
    if subjects == 0
        || periods == 0
        || treatment.len() != cells
        || event.len() != cells
        || at_risk.len() != cells
        || treatment_probability.len() != cells
        || censoring_weight.len() != cells
    {
        return Err("trajectory arrays must share one positive subject-period length");
    }
    if !minimum_probability.is_finite() || minimum_probability <= 0.0 || minimum_probability > 0.5 {
        return Err("minimum treatment probability must be finite and in (0, 0.5]");
    }
    let upper = 1.0 - minimum_probability;
    for (&weight, &probability) in censoring_weight.iter().zip(treatment_probability) {
        if !weight.is_finite() || weight <= 0.0 {
            return Err("censoring weights must be finite and positive");
        }
        if !probability.is_finite() || probability < minimum_probability || probability > upper {
            return Err("treatment probabilities must lie inside the declared positivity floor");
        }
    }
    for subject in 0..subjects {
        let mut left = false;
        for period in 0..periods {
            let j = subject * periods + period;
            if left && at_risk[j] {
                return Err("leaving the risk set is permanent");
            }
            if event[j] && !at_risk[j] {
                return Err("an event requires the subject to be at risk");
            }
            if event[j] || !at_risk[j] {
                left = true;
            }
        }
    }
    Ok(())
}

fn period_masses(
    rows: &TrajectoryRows<'_>,
    regime: &[bool],
) -> Result<Vec<PeriodMass>, &'static str> {
    let mut masses = Vec::with_capacity(rows.periods);
    for period in 0..rows.periods {
        let mut risk = 0.0;
        let mut event = 0.0;
        let mut nonevent = 0.0;
        let mut maximum_weight = 0.0_f64;
        for subject in 0..rows.subjects {
            if !followed(rows, regime, subject, period) {
                continue;
            }
            let weight = cumulative_weight(rows, subject, period)?;
            risk += weight;
            if rows.event[subject * rows.periods + period] {
                event += weight;
            } else {
                nonevent += weight;
            }
            maximum_weight = maximum_weight.max(weight);
        }
        if !risk.is_finite() || risk <= 0.0 {
            return Err("a trajectory risk set has no positive weight");
        }
        if !event.is_finite() || !nonevent.is_finite() || event > risk * (1.0 + 1e-12) {
            return Err("a trajectory event weight exceeds its risk weight");
        }
        masses.push(PeriodMass { risk, event, nonevent, maximum_weight });
    }
    Ok(masses)
}

fn followed(rows: &TrajectoryRows<'_>, regime: &[bool], subject: usize, period: usize) -> bool {
    (0..=period).all(|k| {
        let j = subject * rows.periods + k;
        rows.at_risk[j] && rows.treatment[j] == regime[k]
    })
}

fn cumulative_weight(
    rows: &TrajectoryRows<'_>,
    subject: usize,
    period: usize,
) -> Result<f64, &'static str> {
    let mut weight = 1.0;
    for k in 0..=period {
        let j = subject * rows.periods + k;
        weight *= rows.censoring_weight[j] / rows.treatment_probability[j];
    }
    if weight.is_finite() && weight > 0.0 {
        Ok(weight)
    } else {
        Err("a cumulative trajectory weight must be finite and positive")
    }
}

fn product_limit(masses: &[PeriodMass]) -> TrajectorySurvival {
    let mut survival = 1.0;
    let mut restricted_mean = 0.0;
    let mut minimum_risk_weight = f64::INFINITY;
    let mut maximum_weight = 0.0_f64;
    for mass in masses {
        restricted_mean += survival;
        survival *= 1.0 - mass.event / mass.risk;
        minimum_risk_weight = minimum_risk_weight.min(mass.risk);
        maximum_weight = maximum_weight.max(mass.maximum_weight);
    }
    TrajectorySurvival { survival, restricted_mean, minimum_risk_weight, maximum_weight }
}

fn posterior_means(masses: &[PeriodMass], prior: HazardPrior) -> (f64, f64) {
    let mut survival = 1.0;
    let mut restricted_mean = 0.0;
    for mass in masses {
        restricted_mean += survival;
        let stay = (prior.nonevents + mass.nonevent) / (prior.events + prior.nonevents + mass.risk);
        survival *= stay;
    }
    (survival, restricted_mean)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn near(left: f64, right: f64) {
        assert!((left - right).abs() <= 1e-12, "{left} != {right}");
    }

    struct Fixture {
        treatment: Vec<bool>,
        event: Vec<bool>,
        at_risk: Vec<bool>,
        treatment_probability: Vec<f64>,
        censoring_weight: Vec<f64>,
    }

    fn fixture() -> Fixture {
        Fixture {
            treatment: vec![true, true, true, true, false, false, false, false, true, false],
            event: vec![false, true, false, false, true, false, false, false, false, false],
            at_risk: vec![true, true, true, true, true, false, true, true, true, true],
            treatment_probability: vec![0.5, 0.5, 0.25, 0.5, 0.5, 0.5, 0.5, 0.25, 0.5, 0.5],
            censoring_weight: vec![1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0],
        }
    }

    fn rows(fixture: &Fixture) -> TrajectoryRows<'_> {
        TrajectoryRows {
            subjects: 5,
            periods: 2,
            treatment: &fixture.treatment,
            event: &fixture.event,
            at_risk: &fixture.at_risk,
            treatment_probability: &fixture.treatment_probability,
            censoring_weight: &fixture.censoring_weight,
            minimum_probability: 0.2,
        }
    }

    const ACTIVE: [bool; 2] = [true, true];
    const CONTROL: [bool; 2] = [false, false];

    #[test]
    fn weighted_product_limit_recovers_the_hand_computed_contrast() {
        // Active period 0 also includes subject 4 (weight 2, then the trajectory splits),
        // so the risk weight is 2 + 4 + 2 = 8 and there is still no event. Period 1 keeps
        // subjects 0 and 1 only: weights 4 (event) and 16, hazard 1/5, survival 4/5.
        // Control period 0 is subjects 2 and 3: weights 2 (event) and 4, hazard 1/3.
        // Period 1 keeps subject 3 at weight 16, so survival stays 2/3.
        let data = fixture();
        let contrast = trajectory_survival_contrast(&rows(&data), &ACTIVE, &CONTROL).unwrap();
        near(contrast.active.survival, 0.8);
        near(contrast.control.survival, 2.0 / 3.0);
        near(contrast.survival_difference, 2.0 / 15.0);
        near(contrast.active.restricted_mean, 2.0);
        near(contrast.control.restricted_mean, 5.0 / 3.0);
        near(contrast.restricted_mean_difference, 1.0 / 3.0);
        near(contrast.active.minimum_risk_weight, 8.0);
        near(contrast.control.minimum_risk_weight, 6.0);
        near(contrast.active.maximum_weight, 16.0);
        near(contrast.control.maximum_weight, 16.0);
    }

    #[test]
    fn unequal_probabilities_change_the_weighted_product_limit() {
        let mut data = fixture();
        data.treatment_probability.fill(0.5);
        data.censoring_weight.fill(1.0);
        let contrast = trajectory_survival_contrast(&rows(&data), &ACTIVE, &CONTROL).unwrap();
        near(contrast.active.survival, 0.5);
        near(contrast.control.survival, 0.5);
        near(contrast.survival_difference, 0.0);
    }

    #[test]
    fn a_nonpositive_censoring_weight_is_refused() {
        let mut data = fixture();
        data.censoring_weight[5] = 0.0;
        let zero = trajectory_survival_contrast(&rows(&data), &ACTIVE, &CONTROL).unwrap_err();
        assert_eq!(zero, "censoring weights must be finite and positive");
        data.censoring_weight[5] = -1.0;
        let negative =
            trajectory_hazard_posterior(&rows(&data), &ACTIVE, &CONTROL, HazardPrior::uniform())
                .unwrap_err();
        assert_eq!(negative, "censoring weights must be finite and positive");
    }

    #[test]
    fn a_broken_trajectory_is_refused() {
        let data = fixture();
        let empty =
            trajectory_survival_contrast(&rows(&data), &[false, true], &CONTROL).unwrap_err();
        assert_eq!(empty, "a trajectory risk set has no positive weight");

        let mut returned = fixture();
        returned.at_risk[5] = true;
        let reentered =
            trajectory_survival_contrast(&rows(&returned), &ACTIVE, &CONTROL).unwrap_err();
        assert_eq!(reentered, "leaving the risk set is permanent");

        let mut off_risk = fixture();
        off_risk.event[5] = true;
        let event = trajectory_survival_contrast(&rows(&off_risk), &ACTIVE, &CONTROL).unwrap_err();
        assert_eq!(event, "an event requires the subject to be at risk");

        let mut thin = fixture();
        thin.treatment_probability[0] = 0.1;
        let floor = trajectory_survival_contrast(&rows(&thin), &ACTIVE, &CONTROL).unwrap_err();
        assert_eq!(floor, "treatment probabilities must lie inside the declared positivity floor");
    }

    #[test]
    fn independent_hazard_posterior_shrinks_the_weighted_hazards() {
        // Uniform prior on the same weights. Active period-0 mass is 8 non-events, so the
        // expected survival factor is 9/10 rather than the product-limit's 1. Period 1 is
        // 17/22. Control stays 5/8 then 17/18.
        let data = fixture();
        let posterior =
            trajectory_hazard_posterior(&rows(&data), &ACTIVE, &CONTROL, HazardPrior::uniform())
                .unwrap();
        near(posterior.active_survival_mean, 153.0 / 220.0);
        near(posterior.control_survival_mean, 85.0 / 144.0);
        near(posterior.survival_difference_mean, 833.0 / 7920.0);
        near(posterior.active_restricted_mean, 19.0 / 10.0);
        near(posterior.control_restricted_mean, 13.0 / 8.0);
        near(posterior.restricted_mean_difference_mean, 11.0 / 40.0);
        let contrast = trajectory_survival_contrast(&rows(&data), &ACTIVE, &CONTROL).unwrap();
        assert!((posterior.survival_difference_mean - contrast.survival_difference).abs() > 1e-3);

        let improper = trajectory_hazard_posterior(
            &rows(&data),
            &ACTIVE,
            &CONTROL,
            HazardPrior { events: 0.0, nonevents: 1.0 },
        )
        .unwrap_err();
        assert_eq!(improper, "hazard prior counts must be finite and positive");
    }
}
