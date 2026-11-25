//! Counts a stated support row rests on, recomputed from the stored query and payload.
//!
//! The row checks of [`super::rollout_support`], [`super::stated_section_rows`] and the
//! design-posterior statement read the counts a result states about itself: the units, the
//! cohorts, the periods, the donors, the unbounded sets, the requested replicates and the
//! posterior draws. Wherever the artifact carries the data those counts were formed from, the
//! checks here recount them and refuse a result that states support on a count its own query
//! and payload do not give:
//!
//! * a difference-in-differences rollout section: the rows, the units, the treated and the
//!   never-treated units, every adoption cohort's units and the balanced panel are recounted
//!   from the query's subjects, periods and cohorts, and the requested refits are the query's;
//! * a synthetic-control rollout section: the pre-intervention periods, the donor series, the
//!   adopting units and the fewest pre-adoption periods are recounted from the query's units,
//!   periods and adoptions, the unbounded sets from the section's own effects, and the
//!   requested placebo refits are the query's;
//! * a calendar-time local ratio section: the requested block-bootstrap refits are the
//!   query's;
//! * a design posterior that states a row: the draws are counted in the stored posterior;
//! * a temporal design that states a row: its clusters cannot exceed the query's labels.
//!
//! What the artifact cannot recount stays the producer's statement, because the data rows are
//! not stored: the completed (as opposed to requested) replicates of any resample, the window
//! rows and the first-stage strength of a local ratio, the fitted comparison propensity of a
//! covariate-adjusted comparison, the supplied propensity range and the completed refits of an
//! augmented panel, the rows and the fitted probabilities of an association, of instrument
//! bounds and of a sequential contrast, and the analysed pairs and the realized assignment
//! floor of a temporal design (which decisions had an observed outcome is not stored).
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{BTreeMap, BTreeSet};

use super::{AnalysisResultWire, QuasiRolloutWire, TemporalDesignWire};
use crate::graphless_support::{DESIGN_POSTERIOR_ROW_CODE, TEMPORAL_DESIGN_ROW_CODE};
use crate::query_wire::{
    LocalPolynomialRatioQueryWire, PanelDidQueryWire, SyntheticControlQueryWire,
};
use crate::temporal_design_query_wire::{TemporalAssignmentWire, TemporalDesignQueryWire};
use crate::{CausalQueryWire, IoError};

/// Posterior draws every design-posterior support row was measured with: the executor states
/// such a row only on a posterior of at least this many draws.
const DESIGN_POSTERIOR_ROW_DRAWS: u32 = 2_000;

/// Uncertainty label of the calendar-time local ratio's block-bootstrap interval.
const TEMPORAL_BLOCK_BOOTSTRAP: &str =
    "within_side_moving_block_bootstrap_refit_iqr_se_student_t_interval";

/// Refuse a result that states support on a count its stored query and payload do not give.
///
/// A result that states no support is accepted as written. Runs after the row checks, so a
/// stated count below its row is refused there first.
pub(super) fn validate_recomputed_support(result: &AnalysisResultWire) -> Result<(), IoError> {
    let refuse = |what: &str| {
        Err(IoError::Convert(format!(
            "{what} states support on a count its stored query and payload do not give"
        )))
    };
    if result.diagnostics.iter().any(|diagnostic| diagnostic.code == DESIGN_POSTERIOR_ROW_CODE)
        && !design_posterior_draws_hold(result)?
    {
        return refuse("a design posterior");
    }
    if let Some(section) = result.quasi_rollout.as_ref().filter(|section| section.supported) {
        let holds = match &result.query {
            CausalQueryWire::PanelDid(query) => did_rollout_counts_hold(query, section),
            CausalQueryWire::SyntheticControl(query) => {
                synthetic_rollout_counts_hold(query, section)
            }
            CausalQueryWire::LocalPolynomialRatio(query) => {
                local_ratio_replicates_hold(query, section)
            }
            _ => false,
        };
        if !holds {
            return refuse("a rollout section");
        }
    }
    if result.diagnostics.iter().any(|diagnostic| diagnostic.code == TEMPORAL_DESIGN_ROW_CODE)
        && !matches!(
            (&result.query, &result.temporal_design),
            (CausalQueryWire::TemporalDesign(query), Some(section))
                if temporal_clusters_within_query(query, section)
        )
    {
        return refuse("a temporal design");
    }
    Ok(())
}

/// Whether the section's named diagnostic is exactly `count`.
fn states_count(section: &QuasiRolloutWire, name: &str, count: usize) -> bool {
    section.diagnostic(name).is_some_and(|stated| (stated - count as f64).abs() <= 0.0)
}

/// Whether the stored posterior holds the draws its stated rows were measured with: at least
/// [`DESIGN_POSTERIOR_ROW_DRAWS`], and no stated row counts more draws behind its interval
/// than the posterior stores.
fn design_posterior_draws_hold(result: &AnalysisResultWire) -> Result<bool, IoError> {
    let Some(bytes) = &result.posterior_artifact else {
        return Ok(false);
    };
    let posterior = crate::decode_posterior_meta_from_bytes(bytes)?;
    let stored = u64::from(posterior.n_draws);
    Ok(posterior.n_draws >= DESIGN_POSTERIOR_ROW_DRAWS
        && result
            .diagnostics
            .iter()
            .filter(|diagnostic| diagnostic.code == DESIGN_POSTERIOR_ROW_CODE)
            .all(|diagnostic| stated_resamples(&diagnostic.fields) <= stored))
}

/// The draws a design-posterior statement counts behind its interval; zero when it states none.
fn stated_resamples(fields: &[(String, String)]) -> u64 {
    fields
        .iter()
        .find(|(name, _)| name == "interval_resamples")
        .map_or(0, |(_, value)| value.parse::<u64>().unwrap_or(u64::MAX))
}

/// Whether the counts a supported difference-in-differences rollout section states are the
/// query's: every supplied row read, the units, the treated and the never-treated units, each
/// adoption cohort's units (zero is never treated), every unit observed in every period, and
/// the requested refits. A unit's cohort is its first row's; the estimator refuses a cohort
/// that changes within a unit.
fn did_rollout_counts_hold(query: &PanelDidQueryWire, section: &QuasiRolloutWire) -> bool {
    let rows = query.subjects.len();
    if query.periods.len() != rows || query.cohorts.len() != rows {
        return false;
    }
    let mut cohort_of = BTreeMap::<&str, i64>::new();
    let mut periods_of = BTreeMap::<&str, BTreeSet<i64>>::new();
    let mut calendar = BTreeSet::new();
    for ((subject, period), cohort) in query.subjects.iter().zip(&query.periods).zip(&query.cohorts)
    {
        cohort_of.entry(subject.as_str()).or_insert(*cohort);
        periods_of.entry(subject.as_str()).or_default().insert(*period);
        calendar.insert(*period);
    }
    let mut cohort_units = BTreeMap::<i64, usize>::new();
    for cohort in cohort_of.values().filter(|cohort| **cohort > 0) {
        *cohort_units.entry(*cohort).or_default() += 1;
    }
    let treated = cohort_units.values().sum::<usize>();
    let never_treated = cohort_of.values().filter(|cohort| **cohort == 0).count();
    let stated_cohorts = section
        .diagnostics
        .iter()
        .filter(|(name, _)| name.starts_with("cohort_") && name.ends_with("_units"))
        .count();
    periods_of.values().all(|observed| observed.len() == calendar.len())
        && states_count(section, "rows_supplied", rows)
        && states_count(section, "rows_read", rows)
        && states_count(section, "units", cohort_of.len())
        && states_count(section, "treated_units", treated)
        && states_count(section, "never_treated_units", never_treated)
        && stated_cohorts == cohort_units.len()
        && cohort_units
            .iter()
            .all(|(cohort, units)| states_count(section, &format!("cohort_{cohort}_units"), *units))
        && query
            .resampling
            .as_ref()
            .is_some_and(|resampling| resampling.replicates == section.replicates_requested)
}

/// Whether the counts a supported synthetic-control rollout section states are the query's.
///
/// Conformal sets: the calendar periods before the intervention, the units other than the
/// treated one, and the effects published without a set (a set reaching the searched range is
/// published as absent). Placebo variance: the adopting units, the units that never adopt,
/// the fewest calendar periods before an adoption, and the requested placebo refits.
fn synthetic_rollout_counts_hold(
    query: &SyntheticControlQueryWire,
    section: &QuasiRolloutWire,
) -> bool {
    let Some(rollout) = &query.rollout else {
        return false;
    };
    let calendar: BTreeSet<i64> = query.periods.iter().copied().collect();
    let units: BTreeSet<&str> = query.units.iter().map(String::as_str).collect();
    match section.design.as_str() {
        "conformal_synthetic_control" => {
            let pre_periods =
                calendar.iter().filter(|period| **period < query.intervention_period).count();
            let unbounded =
                section.effects.iter().filter(|effect| effect.interval.is_none()).count();
            units.contains(query.treated_unit.as_str())
                && states_count(section, "pre_periods", pre_periods)
                && states_count(section, "donors", units.len() - 1)
                && states_count(section, "unbounded_intervals", unbounded)
        }
        "multi_unit_synthetic_control" => {
            let adopting: BTreeSet<&str> =
                rollout.adoptions.iter().map(|(unit, _)| unit.as_str()).collect();
            let fewest_pre_periods = rollout
                .adoptions
                .iter()
                .map(|(_, adoption)| calendar.iter().filter(|period| **period < *adoption).count())
                .min()
                .unwrap_or(0);
            adopting.len() == rollout.adoptions.len()
                && adopting.iter().all(|unit| units.contains(unit))
                && states_count(section, "adopting_units", adopting.len())
                && states_count(section, "never_treated_donors", units.len() - adopting.len())
                && states_count(section, "fewest_pre_periods", fewest_pre_periods)
                && section.replicates_requested == rollout.replicates
        }
        _ => false,
    }
}

/// Whether the refits a supported calendar-time local ratio section requested are the query's.
/// Every other local ratio section has no resample to recount.
fn local_ratio_replicates_hold(
    query: &LocalPolynomialRatioQueryWire,
    section: &QuasiRolloutWire,
) -> bool {
    section.uncertainty != TEMPORAL_BLOCK_BOOTSTRAP
        || query.options.as_ref().and_then(|options| options.temporal_replicates)
            == Some(section.replicates_requested)
}

/// Whether the clusters a temporal design section states fit in the query: no more than its
/// distinct interference clusters (an interference design) or distinct units (every other
/// design). The section counts the clusters that hold an analysed pair, which the artifact
/// cannot recount exactly because it does not store which decisions had an observed outcome.
fn temporal_clusters_within_query(
    query: &TemporalDesignQueryWire,
    section: &TemporalDesignWire,
) -> bool {
    let labels: BTreeSet<&str> = match &query.assignment {
        TemporalAssignmentWire::Interference { clusters, .. } => {
            clusters.iter().map(String::as_str).collect()
        }
        _ => query.units.iter().map(String::as_str).collect(),
    };
    usize::try_from(section.clusters).is_ok_and(|clusters| clusters <= labels.len())
}

#[cfg(test)]
mod tests {
    use super::{
        did_rollout_counts_hold, local_ratio_replicates_hold, stated_resamples,
        synthetic_rollout_counts_hold, temporal_clusters_within_query,
    };
    use crate::analysis_result_artifact::{DesignEffectWire, QuasiRolloutWire, TemporalDesignWire};
    use crate::query_wire::{
        LocalPolynomialRatioQueryWire, PanelDidQueryWire, SyntheticControlQueryWire,
    };
    use crate::rollout_query_wire::{
        DidResamplingWire, LocalRatioOptionsWire, SyntheticRolloutWire,
    };
    use crate::temporal_design_query_wire::{TemporalAssignmentWire, TemporalDesignQueryWire};

    fn effect(kind: &str, interval: Option<[f64; 2]>) -> DesignEffectWire {
        DesignEffectWire {
            kind: kind.to_owned(),
            group: None,
            period: None,
            event_time: None,
            unit: None,
            estimate: 1.0,
            standard_error: None,
            interval,
            band: None,
            treated: 1,
            comparison: 1,
        }
    }

    fn section(
        design: &str,
        uncertainty: &str,
        requested: u32,
        effects: Vec<DesignEffectWire>,
        diagnostics: &[(&str, f64)],
    ) -> QuasiRolloutWire {
        QuasiRolloutWire {
            design: design.to_owned(),
            method: String::new(),
            effects,
            uncertainty: uncertainty.to_owned(),
            calibration: "not_measured".to_owned(),
            replicates_requested: requested,
            replicates_ok: requested,
            replicates_failed: 0,
            critical_value: None,
            diagnostics: diagnostics
                .iter()
                .map(|(name, value)| ((*name).to_owned(), *value))
                .collect(),
            supported: true,
        }
    }

    fn restated(section: &QuasiRolloutWire, name: &str, value: f64) -> QuasiRolloutWire {
        let mut forged = section.clone();
        let entry = forged.diagnostics.iter_mut().find(|(key, _)| key == name).expect(name);
        entry.1 = value;
        forged
    }

    /// Six units over three periods: two adopt at period 2, one at period 3, three never.
    fn staggered() -> PanelDidQueryWire {
        let cohorts = [2_i64, 2, 3, 0, 0, 0];
        let (mut subjects, mut periods, mut cohort_rows) = (vec![], vec![], vec![]);
        for (unit, cohort) in cohorts.iter().enumerate() {
            for period in 1..=3_i64 {
                subjects.push(format!("u{unit}"));
                periods.push(period);
                cohort_rows.push(*cohort);
            }
        }
        PanelDidQueryWire {
            outcome: 0,
            augmented: None,
            repeated_cross_section: false,
            staggered_target: None,
            staggered_event_study: true,
            periods,
            cohorts: cohort_rows.clone(),
            treated: cohort_rows.iter().map(|cohort| *cohort > 0).collect(),
            post: vec![false; subjects.len()],
            clusters: subjects.clone(),
            subjects,
            rollout: None,
            resampling: Some(DidResamplingWire {
                scheme: "cluster_multiplier".to_owned(),
                replicates: 199,
            }),
        }
    }

    const MULTIPLIER: &str = "whole_cluster_multiplier_refit_iqr_se_pointwise_and_sup_t_band";

    fn staggered_section() -> QuasiRolloutWire {
        section(
            "staggered_did",
            MULTIPLIER,
            199,
            vec![effect("overall", Some([0.0, 2.0]))],
            &[
                ("units", 6.0),
                ("treated_units", 3.0),
                ("never_treated_units", 3.0),
                ("clusters", 6.0),
                ("rows_read", 18.0),
                ("rows_supplied", 18.0),
                ("balanced_panel", 1.0),
                ("cohort_2_units", 2.0),
                ("cohort_2_rows", 6.0),
                ("cohort_3_units", 1.0),
                ("cohort_3_units_at_base", 1.0),
            ],
        )
    }

    #[test]
    fn a_rollout_section_must_state_the_counts_of_its_query() {
        let query = staggered();
        let stated = staggered_section();
        assert!(did_rollout_counts_hold(&query, &stated));
        // Each stated count restated away from the query's is refused, upward included.
        for (name, value) in [
            ("units", 200.0),
            ("treated_units", 100.0),
            ("never_treated_units", 100.0),
            ("rows_read", 800.0),
            ("rows_supplied", 800.0),
            ("cohort_2_units", 60.0),
            ("cohort_3_units", 40.0),
        ] {
            assert!(!did_rollout_counts_hold(&query, &restated(&stated, name, value)), "{name}");
        }
        // A cohort left out of the section, or one the query does not have.
        let mut dropped = stated.clone();
        dropped.diagnostics.retain(|(name, _)| name != "cohort_3_units");
        assert!(!did_rollout_counts_hold(&query, &dropped));
        let mut extra = stated.clone();
        extra.diagnostics.push(("cohort_5_units".to_owned(), 50.0));
        assert!(!did_rollout_counts_hold(&query, &extra));
        // More refits than the query declared.
        let mut refits = stated.clone();
        (refits.replicates_requested, refits.replicates_ok) = (999, 999);
        assert!(!did_rollout_counts_hold(&query, &refits));
        // A unit missing a period is not the balanced panel the rows are stated for.
        let mut unbalanced = query.clone();
        for column in [&mut unbalanced.periods, &mut unbalanced.cohorts] {
            column.pop();
        }
        unbalanced.subjects.pop();
        assert!(!did_rollout_counts_hold(&unbalanced, &restated(&stated, "rows_read", 17.0)));
    }

    /// Ten donors and one treated unit over 44 periods, treated from period 40.
    fn one_treated() -> SyntheticControlQueryWire {
        let (mut units, mut periods) = (vec![], vec![]);
        for period in 1..=44_i64 {
            for unit in 0..11 {
                units.push(if unit == 10 { "treated".to_owned() } else { format!("d{unit}") });
                periods.push(period);
            }
        }
        SyntheticControlQueryWire {
            outcome: 0,
            units,
            periods,
            treated_unit: "treated".to_owned(),
            intervention_period: 40,
            difference_in_differences: false,
            uniform_unit_randomization: false,
            sharp_null_effect: None,
            augmentation_ridge: None,
            rollout: Some(SyntheticRolloutWire {
                adoptions: vec![("treated".to_owned(), 40)],
                interval: "conformal".to_owned(),
                replicates: 0,
            }),
            observed_window_support: false,
        }
    }

    #[test]
    fn conformal_counts_are_recounted_from_the_query_and_the_effects() {
        let query = one_treated();
        let effects = |bounded: usize| {
            let mut effects = vec![effect("overall", Some([0.0, 2.0]))];
            effects.extend(
                (0..5).map(|index| effect("period", (index < bounded).then_some([0.0, 2.0]))),
            );
            effects
        };
        let stated = |effects, unbounded: f64| {
            section(
                "conformal_synthetic_control",
                "conformal_test_inversion_refit_under_each_null",
                0,
                effects,
                &[("pre_periods", 39.0), ("donors", 10.0), ("unbounded_intervals", unbounded)],
            )
        };
        let whole = stated(effects(5), 0.0);
        assert!(synthetic_rollout_counts_hold(&query, &whole));
        assert!(!synthetic_rollout_counts_hold(&query, &restated(&whole, "pre_periods", 159.0)));
        assert!(!synthetic_rollout_counts_hold(&query, &restated(&whole, "donors", 40.0)));
        // A set published as absent is an unbounded set, whatever the section states.
        assert!(!synthetic_rollout_counts_hold(&query, &stated(effects(4), 0.0)));
        assert!(synthetic_rollout_counts_hold(&query, &stated(effects(4), 1.0)));
    }

    #[test]
    fn placebo_counts_are_recounted_from_the_adoptions() {
        let mut query = one_treated();
        // Three of the ten donors adopt at periods 21, 24 and 27; the labelled unit is a donor.
        query.rollout = Some(SyntheticRolloutWire {
            adoptions: vec![("d0".to_owned(), 21), ("d1".to_owned(), 24), ("d2".to_owned(), 27)],
            interval: "placebo_variance".to_owned(),
            replicates: 199,
        });
        let stated = section(
            "multi_unit_synthetic_control",
            "placebo_variance_refit_normal_interval_and_per_unit_sup_t_band",
            199,
            vec![effect("overall", Some([0.0, 2.0]))],
            &[("adopting_units", 3.0), ("never_treated_donors", 8.0), ("fewest_pre_periods", 20.0)],
        );
        assert!(synthetic_rollout_counts_hold(&query, &stated));
        for (name, value) in
            [("adopting_units", 5.0), ("never_treated_donors", 40.0), ("fewest_pre_periods", 30.0)]
        {
            assert!(
                !synthetic_rollout_counts_hold(&query, &restated(&stated, name, value)),
                "{name}"
            );
        }
        let mut refits = stated;
        (refits.replicates_requested, refits.replicates_ok) = (999, 999);
        assert!(!synthetic_rollout_counts_hold(&query, &refits));
    }

    #[test]
    fn a_calendar_time_local_ratio_requests_the_refits_of_its_query() {
        let query = |replicates| LocalPolynomialRatioQueryWire {
            outcome: 0,
            treatment: 1,
            running: 2,
            cutoff: 0.0,
            bandwidth: 1.0,
            kink: false,
            options: Some(LocalRatioOptionsWire {
                bandwidth_rule: "declared".to_owned(),
                temporal_replicates: replicates,
                density_test: false,
                balance_covariates: vec![],
            }),
        };
        let bootstrap =
            section("temporal_fuzzy_rd", super::TEMPORAL_BLOCK_BOOTSTRAP, 199, vec![], &[]);
        assert!(local_ratio_replicates_hold(&query(Some(199)), &bootstrap));
        assert!(!local_ratio_replicates_hold(&query(Some(50)), &bootstrap));
        assert!(!local_ratio_replicates_hold(&query(None), &bootstrap));
        // A section without a resample has nothing to recount.
        let analytic = section("fuzzy_rd", "rbc_hc0_delta_normal_fixed_bandwidth", 0, vec![], &[]);
        assert!(local_ratio_replicates_hold(&query(None), &analytic));
    }

    #[test]
    fn a_statement_without_a_readable_draw_count_counts_every_draw() {
        let fields = |value: &str| vec![("interval_resamples".to_owned(), value.to_owned())];
        assert_eq!(stated_resamples(&fields("2000")), 2000);
        assert_eq!(stated_resamples(&fields("many")), u64::MAX);
        assert_eq!(stated_resamples(&[]), 0);
    }

    #[test]
    fn temporal_clusters_cannot_exceed_the_labels_of_the_query() {
        let units: Vec<String> = (0..8).map(|row| format!("u{}", row / 2)).collect();
        let two_arm = TemporalDesignQueryWire {
            outcome: 0,
            units,
            periods: (0..8).map(|row| i64::from(row % 2)).collect(),
            lag: 0,
            assignment: TemporalAssignmentWire::TwoArm {
                assignment: vec![true; 8],
                probability: vec![0.5; 8],
            },
            quantile_level: None,
            min_probability: 0.05,
            outcome_misclassification: None,
        };
        let stated = |clusters: u64| TemporalDesignWire {
            design: "two_arm".to_owned(),
            functional: "mean".to_owned(),
            coordinates: vec!["effect".to_owned()],
            estimate: vec![1.0],
            standard_error: vec![0.1],
            lower: vec![0.8],
            upper: vec![1.2],
            covariance: None,
            band_coordinates: vec![],
            band_critical_value: None,
            band_lower: None,
            band_upper: None,
            holm_adjusted_p: None,
            smoothing_bias_bound: None,
            pairs: 8,
            clusters,
            unpaired: 0,
            minimum_probability: Some(0.5),
            replicates: None,
            draws: None,
            uncertainty: "cluster_robust_ratio_normal".to_owned(),
            calibration: "not_measured".to_owned(),
            supported: false,
        };
        assert!(temporal_clusters_within_query(&two_arm, &stated(4)));
        assert!(!temporal_clusters_within_query(&two_arm, &stated(150)));
        // An interference design counts its interference clusters, not its units.
        let mut interference = two_arm;
        interference.assignment = TemporalAssignmentWire::Interference {
            assignment: vec![true; 8],
            probability: vec![0.5; 8],
            clusters: (0..8).map(|row| format!("c{}", row / 4)).collect(),
        };
        assert!(temporal_clusters_within_query(&interference, &stated(2)));
        assert!(!temporal_clusters_within_query(&interference, &stated(4)));
    }
}
