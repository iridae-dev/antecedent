//! Bayesian cohort-specific event-time contrasts against never-treated subjects.
//!
//! Each contrast is the same estimand as [`crate::staggered_event_study::estimate`]: the
//! difference between a cohort's and the never-treated subjects' outcome change from the cohort's
//! reference period `g - 1` to the contrast period. The posterior is the Normal-Inverse-Gamma
//! posterior of the group coefficient of the change score on `1 + cohort`, with independent
//! subjects and one change-score variance for the contrast's two groups.
//!
//! The premise is parallel untreated trends and no anticipation relative to the never-treated
//! group. Pre-adoption leads are reported beside the post-adoption effects as descriptive
//! posteriors: a lead concentrated near zero does not license parallel trends after adoption.
//! Contrasts share their never-treated subjects, so they are dependent; each posterior here is a
//! pointwise marginal and nothing is claimed simultaneously across the curve.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{BTreeMap, BTreeSet};

use crate::bayesian_panel_did::{DidPrior, fit_nig, require_one_unit_per_cluster, summarize};
use crate::error::EstimationError;
use crate::util::DrawSpec;

/// Seed offset that separates the independent fits of different contrasts.
const CONTRAST_SEED_STRIDE: u64 = 0xA24B_AED4_963E_E407;

/// Posterior of one cohort-specific event-time contrast.
#[derive(Clone, Debug, PartialEq)]
pub struct BayesianEventTimeEffect {
    /// First treated period of the adoption cohort.
    pub cohort: i64,
    /// Calendar period of the contrast.
    pub period: i64,
    /// Calendar period minus adoption cohort; the reference `-1` is omitted.
    pub event_time: i64,
    /// Marginal posterior draws of the contrast.
    pub effect_draws: Vec<f64>,
    /// Posterior mean.
    pub mean: f64,
    /// Posterior standard deviation.
    pub sd: f64,
    /// Equal-tailed 95% lower posterior quantile.
    pub lower: f64,
    /// Equal-tailed 95% upper posterior quantile.
    pub upper: f64,
    /// Subjects in the adoption cohort.
    pub treated_subjects: usize,
    /// Never-treated comparison subjects.
    pub comparison_subjects: usize,
}

type Panel<'a> = BTreeMap<&'a str, (i64, &'a str, BTreeMap<i64, f64>)>;

fn balanced_panel<'a>(
    outcomes: &[f64],
    subjects: &'a [impl AsRef<str>],
    periods: &[i64],
    cohorts: &[i64],
    clusters: &'a [impl AsRef<str>],
) -> Result<(Panel<'a>, BTreeSet<i64>), EstimationError> {
    let n = outcomes.len();
    if n == 0
        || subjects.len() != n
        || periods.len() != n
        || cohorts.len() != n
        || clusters.len() != n
    {
        return Err(crate::util::refuse(
            antecedent_core::reason_code!("invalid_argument"),
            "bayesian_event_study_shape",
            "bayesian event study needs equal non-empty outcome, subject, period, cohort and cluster vectors. Next attempt: supply equal-length non-empty outcome, subject, period, cohort and cluster columns",
        ));
    }
    let mut units: Panel<'a> = BTreeMap::new();
    let mut all_periods = BTreeSet::new();
    for i in 0..n {
        let subject = subjects[i].as_ref();
        let cluster = clusters[i].as_ref();
        if subject.trim().is_empty()
            || cluster.trim().is_empty()
            || !outcomes[i].is_finite()
            || periods[i] <= 0
            || cohorts[i] < 0
        {
            return Err(crate::util::refuse(
                antecedent_core::reason_code!("invalid_argument"),
                "bayesian_event_study_labels",
                "bayesian event study needs finite outcomes, non-empty ids, positive periods and nonnegative cohorts. Next attempt: give every row a finite outcome, a non-empty subject and cluster, a positive period and a nonnegative cohort",
            ));
        }
        all_periods.insert(periods[i]);
        let unit = units.entry(subject).or_insert_with(|| (cohorts[i], cluster, BTreeMap::new()));
        if unit.0 != cohorts[i] || unit.1 != cluster {
            return Err(crate::util::refuse(
                antecedent_core::reason_code!("invalid_argument"),
                "bayesian_event_study_unstable_unit",
                "adoption cohort and cluster must remain constant within subject. Next attempt: keep the adoption cohort and cluster constant within a subject",
            ));
        }
        if unit.2.insert(periods[i], outcomes[i]).is_some() {
            return Err(crate::util::refuse(
                antecedent_core::reason_code!("invalid_argument"),
                "bayesian_event_study_duplicate_cell",
                "each subject must have exactly one observation per period. Next attempt: keep one observation per subject and period",
            ));
        }
    }
    let balanced_message =
        "bayesian event study requires a balanced panel with at least two common periods";
    if all_periods.len() < 2 {
        return Err(crate::util::refuse(
            antecedent_core::reason_code!("invalid_argument"),
            "bayesian_event_study_periods",
            &format!(
                "{balanced_message}. Next attempt: a table with the periods before and after \
                 each adoption, or the static randomized effect for one period"
            ),
        ));
    }
    if units.values().any(|unit| all_periods.iter().any(|period| !unit.2.contains_key(period))) {
        return Err(crate::util::refuse(
            antecedent_core::reason_code!("missing_policy_required"),
            "bayesian_event_study_unbalanced",
            &format!(
                "{balanced_message}. Next attempt: the available-case frequentist contrast, or a \
                 missingness model"
            ),
        ));
    }
    let unit_clusters: Vec<&str> = units.values().map(|unit| unit.1).collect();
    require_one_unit_per_cluster(&unit_clusters)?;
    Ok((units, all_periods))
}

fn change_score(unit: &(i64, &str, BTreeMap<i64, f64>), period: i64, baseline: i64) -> f64 {
    unit.2.get(&period).copied().unwrap_or(f64::NAN)
        - unit.2.get(&baseline).copied().unwrap_or(f64::NAN)
}

/// Posterior of every cohort-by-period contrast against never-treated subjects.
///
/// Requires a balanced panel, never-treated subjects (cohort 0), an observed reference period
/// `g - 1` and adoption period for every cohort, at least two subjects in each contrast group,
/// and one cluster per subject.
pub fn fit_bayesian_event_study(
    outcomes: &[f64],
    subjects: &[impl AsRef<str>],
    periods: &[i64],
    cohorts: &[i64],
    clusters: &[impl AsRef<str>],
    prior: DidPrior,
    sampling: DrawSpec,
) -> Result<Vec<BayesianEventTimeEffect>, EstimationError> {
    let (units, all_periods) = balanced_panel(outcomes, subjects, periods, cohorts, clusters)?;
    let cohorts_present: BTreeSet<i64> = units.values().map(|unit| unit.0).collect();
    if !cohorts_present.contains(&0) {
        return Err(crate::util::refuse(
            antecedent_core::reason_code!("invalid_argument"),
            "bayesian_event_study_no_controls",
            "bayesian event study requires never-treated controls (cohort 0). Next attempt: include never-treated subjects (cohort 0), or the not-yet-treated frequentist comparison",
        ));
    }
    let controls: Vec<_> = units.values().filter(|unit| unit.0 == 0).collect();
    let mut effects = Vec::new();
    for cohort in cohorts_present.iter().copied().filter(|cohort| *cohort > 0) {
        let baseline = cohort - 1;
        if !all_periods.contains(&baseline) || !all_periods.contains(&cohort) {
            return Err(crate::util::refuse(
                antecedent_core::reason_code!("invalid_argument"),
                "bayesian_event_study_baseline",
                "each adoption cohort needs an observed immediately pre-treatment baseline and adoption period. Next attempt: observe the period immediately before and the period of each adoption cohort, or drop the cohort",
            ));
        }
        let treated: Vec<_> = units.values().filter(|unit| unit.0 == cohort).collect();
        if treated.len() < 2 || controls.len() < 2 {
            return Err(crate::util::refuse(
                antecedent_core::reason_code!("arm_not_populated"),
                "bayesian_did_arm_support",
                "each contrast group needs at least two subjects. Next attempt: more subjects in \
                 the thin cohort or never-treated group, or a coarser cohort definition",
            ));
        }
        for period in all_periods.iter().copied().filter(|period| *period != baseline) {
            let y: Vec<f64> = controls
                .iter()
                .chain(&treated)
                .map(|unit| change_score(unit, period, baseline))
                .collect();
            let group: Vec<f64> =
                controls.iter().map(|_| 0.0).chain(treated.iter().map(|_| 1.0)).collect();
            let columns = [vec![1.0; y.len()], group];
            let seed =
                sampling.seed.wrapping_add(CONTRAST_SEED_STRIDE.wrapping_mul(effects.len() as u64));
            let fit = fit_nig(&columns, &y, prior, DrawSpec::new(sampling.draws, seed))?;
            let effect_draws = fit.draws.column(1)?.to_vec();
            let (mean, sd, lower, upper) = summarize(&effect_draws);
            effects.push(BayesianEventTimeEffect {
                cohort,
                period,
                event_time: period - cohort,
                effect_draws,
                mean,
                sd,
                lower,
                upper,
                treated_subjects: treated.len(),
                comparison_subjects: controls.len(),
            });
        }
    }
    if effects.is_empty() {
        return Err(crate::util::refuse(
            antecedent_core::reason_code!("invalid_argument"),
            "bayesian_event_study_empty",
            "no event-time contrasts are available. Next attempt: include at least one adoption cohort with an observed pre-period and post-period",
        ));
    }
    Ok(effects)
}

#[cfg(test)]
mod tests;
