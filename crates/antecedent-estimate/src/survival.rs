//! Native risk-set summaries for randomized survival outcomes.
// SPDX-License-Identifier: MIT OR Apache-2.0

pub use antecedent_stats::survival::{
    KnownCensoring, RandomizedSurvivalSummary, SurvivalBootstrapIntervals, SurvivalDifferenceBand,
    SurvivalEndpoint, SurvivalRows, randomized_survival_bootstrap_difference_band,
    randomized_survival_bootstrap_intervals, randomized_survival_ipcw_summary,
    randomized_survival_summary,
};
pub(crate) use antecedent_stats::survival::{bootstrap_support_count, percentile_95};

#[cfg(test)]
use crate::splitmix::splitmix64;
#[cfg(test)]
use antecedent_stats::survival::for_each_resampled_subject;

#[cfg(test)]
mod tests {
    #![cfg_attr(
        test,
        allow(
            clippy::float_cmp,
            reason = "tests assert exact deterministic estimates and match exact event times"
        )
    )]
    use super::*;

    #[test]
    fn bootstrap_support_count_preserves_thresholds_and_errors() {
        let check = |draws, requested, minimum| {
            bootstrap_support_count(draws, requested, minimum, "overflow", "support")
        };
        assert_eq!(check(199, 199, 199), Ok(199));
        assert_eq!(check(198, 199, 199), Err("support"));
        assert_eq!(check(900, 1000, 199), Ok(900));
        assert_eq!(check(899, 1000, 199), Err("support"));
        assert_eq!(check(399, 399, 399), Ok(399));
        assert_eq!(check(398, 399, 399), Err("support"));
        if let Ok(overflow) = usize::try_from(u64::from(u32::MAX) + 1) {
            assert_eq!(check(overflow, 199, 199), Err("overflow"));
        }
    }

    #[test]
    fn subject_draws_keep_arm_order_and_stream_position() {
        let arms = [vec![10, 20], vec![30, 40, 50]];
        let mut seed = 0;
        let mut draws = Vec::new();
        for_each_resampled_subject(&arms, &mut seed, |index, treated| {
            draws.push((index, treated));
        });
        let expected = if usize::BITS == 32 {
            [(20, false), (10, false), (30, true), (30, true), (40, true)]
        } else {
            [(20, false), (10, false), (40, true), (40, true), (40, true)]
        };
        assert_eq!(draws, expected);
        assert_eq!(seed, 0x1715_609F_7C74_6C69);
    }

    #[test]
    #[allow(
        clippy::too_many_lines,
        reason = "one fixture exercised end to end: build the arms, bootstrap twice for determinism, then refuse thin arms"
    )]
    fn subject_bootstrap_known_truth_deterministic_and_refuses_thin_arms() {
        let mut durations = vec![3.0; 80];
        let mut events = vec![0; 80];
        let treated = (0..80).map(|i| i >= 40).collect::<Vec<_>>();
        for i in 0..40 {
            if i % 5 == 0 {
                durations[i] = 1.0;
                events[i] = 1;
            }
        }
        for i in 40..80 {
            if i % 10 == 0 {
                durations[i] = 1.0;
                events[i] = 1;
            }
        }
        let first = randomized_survival_bootstrap_intervals(
            &SurvivalRows {
                duration: &durations,
                event_code: &events,
                treated: &treated,
                delayed_entry: None,
            },
            None,
            3.0,
            SurvivalEndpoint::Survival,
            399,
            17,
        )
        .unwrap();
        let repeated = randomized_survival_bootstrap_intervals(
            &SurvivalRows {
                duration: &durations,
                event_code: &events,
                treated: &treated,
                delayed_entry: None,
            },
            None,
            3.0,
            SurvivalEndpoint::Survival,
            399,
            17,
        )
        .unwrap();
        assert_eq!(first, repeated);
        assert_eq!(first.replicates_ok, 399);
        assert!(first.rmst_difference.unwrap()[0] <= 0.2);
        assert!(first.rmst_difference.unwrap()[1] >= 0.2);
        assert!(
            randomized_survival_bootstrap_intervals(
                &SurvivalRows {
                    duration: &durations[..8],
                    event_code: &events[..8],
                    treated: &treated[..8],
                    delayed_entry: None
                },
                None,
                3.0,
                SurvivalEndpoint::Survival,
                399,
                17
            )
            .is_err()
        );
        let short_censoring = vec![1.0; 79];
        assert_eq!(
            randomized_survival_bootstrap_intervals(
                &SurvivalRows {
                    duration: &durations,
                    event_code: &events,
                    treated: &treated,
                    delayed_entry: None
                },
                Some(&KnownCensoring {
                    times: &[3.0],
                    survival: &short_censoring,
                    minimum_probability: 0.05
                }),
                3.0,
                SurvivalEndpoint::Survival,
                399,
                17
            ),
            Err("survival bootstrap subject arrays must be row-aligned")
        );
        assert!(
            randomized_survival_bootstrap_intervals(
                &SurvivalRows {
                    duration: &durations,
                    event_code: &events,
                    treated: &treated,
                    delayed_entry: None
                },
                None,
                3.0,
                SurvivalEndpoint::Survival,
                198,
                17
            )
            .is_err()
        );
        // Delayed entry is now a supported, separately calibrated bootstrap route
        // (see delayed_entry_subject_bootstrap_covers_left_truncated_survival_truth),
        // so supplying entry times is accepted rather than refused.
        assert!(
            randomized_survival_bootstrap_intervals(
                &SurvivalRows {
                    duration: &durations,
                    event_code: &events,
                    treated: &treated,
                    delayed_entry: Some(&vec![0.0; 80])
                },
                None,
                3.0,
                SurvivalEndpoint::Survival,
                399,
                17
            )
            .is_ok()
        );
    }

    #[test]
    fn competing_risk_bootstrap_carries_fixed_known_g_with_subjects() {
        let mut durations = vec![3.0; 80];
        let mut events = vec![0; 80];
        let treated = (0..80).map(|i| i >= 40).collect::<Vec<_>>();
        let mut g = Vec::with_capacity(80 * 3);
        for i in 0..80 {
            if i % 5 == 0 {
                durations[i] = 1.0;
                events[i] = 1;
            } else if i % 7 == 0 {
                durations[i] = 1.0;
                events[i] = 2;
            }
            g.extend_from_slice(&[1.0, if i % 2 == 0 { 0.8 } else { 0.9 }, 0.7]);
        }
        let result = randomized_survival_bootstrap_intervals(
            &SurvivalRows {
                duration: &durations,
                event_code: &events,
                treated: &treated,
                delayed_entry: None,
            },
            Some(&KnownCensoring {
                times: &[0.0, 1.0, 3.0],
                survival: &g,
                minimum_probability: 0.01,
            }),
            3.0,
            SurvivalEndpoint::CumulativeIncidence { target_cause: 1 },
            399,
            19,
        )
        .unwrap();
        assert!(result.rmst_difference.is_none());
        assert_eq!(result.replicates_ok, 399);
        assert!(result.difference_at_tau[0] <= 0.0 && result.difference_at_tau[1] >= 0.0);
    }

    #[test]
    fn repeated_sampling_pointwise_bootstrap_covers_simple_survival_truth() {
        // Two-arm discrete event-time law: P(T=1) is .30 versus .15, otherwise
        // T>tau=3. The true RMST and S(3) differences are +.30 and +.15.
        let mut state = 0xBE22_D4E5_95D2_6B63;
        let mut covered_rmst = 0;
        let mut covered_tau = 0;
        for trial in 0..400 {
            let mut durations = Vec::with_capacity(240);
            let mut events = Vec::with_capacity(240);
            let mut treated = Vec::with_capacity(240);
            for i in 0..240 {
                let arm = i >= 120;
                let u = crate::splitmix::splitmix64_unit(&mut state);
                let failed = u < if arm { 0.15 } else { 0.30 };
                durations.push(if failed { 1.0 } else { 3.0 });
                events.push(i64::from(failed));
                treated.push(arm);
            }
            let intervals = randomized_survival_bootstrap_intervals(
                &SurvivalRows {
                    duration: &durations,
                    event_code: &events,
                    treated: &treated,
                    delayed_entry: None,
                },
                None,
                3.0,
                SurvivalEndpoint::Survival,
                299,
                trial + 700,
            )
            .unwrap();
            let rmst = intervals.rmst_difference.unwrap();
            covered_rmst += u32::from(rmst[0] <= 0.30 && 0.30 <= rmst[1]);
            let tau = intervals.difference_at_tau;
            covered_tau += u32::from(tau[0] <= 0.15 && 0.15 <= tau[1]);
        }
        // At p=.95, 400 trials have a binomial standard error about .011.
        // The .90 floor allows Monte Carlo fluctuation while rejecting material
        // undercoverage for this declared data-generating law.
        assert!(covered_rmst >= 360, "RMST coverage: {covered_rmst}/400");
        assert!(covered_tau >= 360, "S(tau) coverage: {covered_tau}/400");
    }

    #[test]
    fn randomized_survival_scalar_bootstrap_support_boundary_covers_known_truth() {
        // Independent randomized subjects, 120 realized assignments per arm,
        // and the minimum 299 bootstrap draws accepted by this matrix row.
        // P(T=1)=.30 in control and .15 in treatment; all other T exceed
        // tau=3. Thus the known truth for S(3) and RMST differences is
        // .15 and .30 respectively.
        const REPLICATIONS: usize = 2_000;
        let mut state = 0xB0A1_DA7A_951A_0011_u64;
        let mut covered_rmst = 0;
        let mut covered_tau = 0;
        for trial in 0..REPLICATIONS {
            let mut duration = Vec::with_capacity(240);
            let mut event = Vec::with_capacity(240);
            let mut treated = Vec::with_capacity(240);
            for arm in [false, true] {
                for _ in 0..120 {
                    let u = crate::splitmix::splitmix64_unit(&mut state);
                    let failed = u < if arm { 0.15 } else { 0.30 };
                    duration.push(if failed { 1.0 } else { 3.0 });
                    event.push(i64::from(failed));
                    treated.push(arm);
                }
            }
            let fit = randomized_survival_bootstrap_intervals(
                &SurvivalRows {
                    duration: &duration,
                    event_code: &event,
                    treated: &treated,
                    delayed_entry: None,
                },
                None,
                3.0,
                SurvivalEndpoint::Survival,
                299,
                trial as u64 + 37_000,
            )
            .unwrap();
            assert_eq!(fit.replicates_ok, 299);
            let rmst = fit.rmst_difference.unwrap();
            let tau = fit.difference_at_tau;
            covered_rmst += usize::from(rmst[0] <= 0.30 && 0.30 <= rmst[1]);
            covered_tau += usize::from(tau[0] <= 0.15 && 0.15 <= tau[1]);
        }
        eprintln!(
            "graphless survival scalar coverage: RMST {covered_rmst}/2000, S(tau) {covered_tau}/2000"
        );
        assert!((1_860..=1_970).contains(&covered_rmst), "RMST coverage {covered_rmst}/2000");
        assert!((1_860..=1_970).contains(&covered_tau), "S(3) coverage {covered_tau}/2000");
    }

    #[test]
    fn delayed_entry_scalar_bootstrap_support_boundary_covers_known_truth() {
        // Delayed-entry (left-truncated) known-truth coverage gate for the
        // graphless support. Entry is independent of the source event time; a
        // subject entering at one with an event at one is unobserved. The two
        // arm laws imply S(3) and RMST differences of .15 and .25. Sampling is
        // repeated at the supported 160 observed subjects per arm and the
        // minimum 299 accepted subject-bootstrap draws.
        const REPLICATIONS: usize = 2_000;
        let mut state = 0xD3E1_A7E0_2000_6B63_u64;
        let mut covered_rmst = 0;
        let mut covered_tau = 0;
        for trial in 0..REPLICATIONS {
            let mut durations = Vec::with_capacity(320);
            let mut events = Vec::with_capacity(320);
            let mut treated = Vec::with_capacity(320);
            let mut entries = Vec::with_capacity(320);
            for arm in [false, true] {
                let mut observed = 0;
                while observed < 160 {
                    let entry = (splitmix64(&mut state) & 1) as f64;
                    let event_u = crate::splitmix::splitmix64_unit(&mut state);
                    let (first, second) = if arm { (0.20, 0.15) } else { (0.30, 0.20) };
                    let (exit, event) = if event_u < first {
                        (1.0, 1)
                    } else if event_u < first + second {
                        (2.0, 1)
                    } else {
                        (3.0, 0)
                    };
                    if entry >= exit {
                        continue;
                    }
                    durations.push(exit);
                    events.push(event);
                    treated.push(arm);
                    entries.push(entry);
                    observed += 1;
                }
            }
            let fit = randomized_survival_bootstrap_intervals(
                &SurvivalRows {
                    duration: &durations,
                    event_code: &events,
                    treated: &treated,
                    delayed_entry: Some(&entries),
                },
                None,
                3.0,
                SurvivalEndpoint::Survival,
                299,
                trial as u64 + 41_000,
            )
            .unwrap();
            assert_eq!(fit.replicates_ok, 299);
            let rmst = fit.rmst_difference.unwrap();
            let tau = fit.difference_at_tau;
            covered_rmst += usize::from(rmst[0] <= 0.25 && 0.25 <= rmst[1]);
            covered_tau += usize::from(tau[0] <= 0.15 && 0.15 <= tau[1]);
        }
        eprintln!(
            "delayed-entry survival scalar coverage: RMST {covered_rmst}/2000, S(tau) {covered_tau}/2000"
        );
        assert!(
            (1_860..=1_970).contains(&covered_rmst),
            "delayed-entry RMST coverage {covered_rmst}/2000"
        );
        assert!(
            (1_860..=1_970).contains(&covered_tau),
            "delayed-entry S(3) coverage {covered_tau}/2000"
        );
    }

    #[test]
    fn fixed_g_ipcw_scalar_bootstrap_support_boundary_covers_known_truth() {
        // Fixed-known-G IPCW known-truth coverage gate for the graphless
        // support, with no delayed entry. Event risk is randomized and
        // independent of the heterogeneous, subject-fixed censoring survival G.
        // The source-population S(3) and RMST differences are .15 and .30.
        // Sampling is repeated at 160 subjects per arm and the minimum 299
        // accepted subject-bootstrap draws.
        const REPLICATIONS: usize = 2_000;
        let mut state = 0xE043_DA75_2000_9839_u64;
        let mut covered_rmst = 0;
        let mut covered_tau = 0;
        for trial in 0..REPLICATIONS {
            let mut durations = Vec::with_capacity(320);
            let mut events = Vec::with_capacity(320);
            let mut treated = Vec::with_capacity(320);
            let mut g = Vec::with_capacity(320 * 4);
            for i in 0..320 {
                let arm = i >= 160;
                let keep_probability = if i % 2 == 0 { 0.9 } else { 0.7 };
                let censor_u = crate::splitmix::splitmix64_unit(&mut state);
                let event_u = crate::splitmix::splitmix64_unit(&mut state);
                let censored = censor_u >= keep_probability;
                let failed = event_u < if arm { 0.15 } else { 0.30 };
                durations.push(if censored {
                    0.5
                } else if failed {
                    1.0
                } else {
                    3.0
                });
                events.push(i64::from(!censored && failed));
                treated.push(arm);
                g.extend_from_slice(&[1.0, keep_probability, keep_probability, keep_probability]);
            }
            let fit = randomized_survival_bootstrap_intervals(
                &SurvivalRows {
                    duration: &durations,
                    event_code: &events,
                    treated: &treated,
                    delayed_entry: None,
                },
                Some(&KnownCensoring {
                    times: &[0.0, 0.5, 1.0, 3.0],
                    survival: &g,
                    minimum_probability: 0.01,
                }),
                3.0,
                SurvivalEndpoint::Survival,
                299,
                trial as u64 + 43_000,
            )
            .unwrap();
            assert_eq!(fit.replicates_ok, 299);
            let rmst = fit.rmst_difference.unwrap();
            let tau = fit.difference_at_tau;
            covered_rmst += usize::from(rmst[0] <= 0.30 && 0.30 <= rmst[1]);
            covered_tau += usize::from(tau[0] <= 0.15 && 0.15 <= tau[1]);
        }
        eprintln!(
            "fixed-G IPCW survival scalar coverage: RMST {covered_rmst}/2000, S(tau) {covered_tau}/2000"
        );
        assert!(
            (1_860..=1_970).contains(&covered_rmst),
            "fixed-G RMST coverage {covered_rmst}/2000"
        );
        assert!((1_860..=1_970).contains(&covered_tau), "fixed-G S(3) coverage {covered_tau}/2000");
    }

    #[test]
    fn delayed_entry_fixed_g_scalar_bootstrap_support_boundary_covers_known_truth() {
        // Combined delayed-entry and fixed-known-G IPCW known-truth coverage
        // gate. Independent entry, randomized treatment, and subject-fixed
        // stratum-specific censoring survival. The source-population S(3) and
        // RMST differences are .15 and .25. Sampling is repeated at 200
        // observed subjects per arm and the minimum 299 accepted draws.
        const REPLICATIONS: usize = 2_000;
        let mut state = 0xD3E1_F1E0_2000_6B63_u64;
        let times = [0.0, 1.0, 1.5, 2.0, 2.5, 3.0];
        let mut covered_rmst = 0;
        let mut covered_tau = 0;
        for trial in 0..REPLICATIONS {
            let mut durations = Vec::with_capacity(400);
            let mut events = Vec::with_capacity(400);
            let mut treated = Vec::with_capacity(400);
            let mut entries = Vec::with_capacity(400);
            let mut g = Vec::with_capacity(400 * times.len());
            for arm in [false, true] {
                let mut observed = 0;
                while observed < 200 {
                    let entry = (splitmix64(&mut state) & 1) as f64;
                    let u = crate::splitmix::splitmix64_unit(&mut state);
                    let (first, second) = if arm { (0.20, 0.15) } else { (0.30, 0.20) };
                    let (event_time, event): (f64, i64) = if u < first {
                        (1.0, 1)
                    } else if u < first + second {
                        (2.0, 1)
                    } else {
                        (3.0, 0)
                    };
                    let high_censoring = splitmix64(&mut state) & 1 == 1;
                    let censor_hazard = if high_censoring { 0.20 } else { 0.10 };
                    let c1 = crate::splitmix::splitmix64_unit(&mut state);
                    let c2 = crate::splitmix::splitmix64_unit(&mut state);
                    let censor_time = if c1 < censor_hazard {
                        1.5
                    } else if c2 < censor_hazard {
                        2.5
                    } else {
                        3.0
                    };
                    let exit = event_time.min(censor_time);
                    if entry >= exit {
                        continue;
                    }
                    durations.push(exit);
                    events.push(if event_time <= censor_time { event } else { 0 });
                    treated.push(arm);
                    entries.push(entry);
                    g.extend_from_slice(&[
                        1.0,
                        1.0,
                        1.0,
                        1.0 - censor_hazard,
                        1.0 - censor_hazard,
                        (1.0 - censor_hazard).powi(2),
                    ]);
                    observed += 1;
                }
            }
            let fit = randomized_survival_bootstrap_intervals(
                &SurvivalRows {
                    duration: &durations,
                    event_code: &events,
                    treated: &treated,
                    delayed_entry: Some(&entries),
                },
                Some(&KnownCensoring { times: &times, survival: &g, minimum_probability: 0.01 }),
                3.0,
                SurvivalEndpoint::Survival,
                299,
                trial as u64 + 45_000,
            )
            .unwrap();
            assert_eq!(fit.replicates_ok, 299);
            let rmst = fit.rmst_difference.unwrap();
            let tau = fit.difference_at_tau;
            covered_rmst += usize::from(rmst[0] <= 0.25 && 0.25 <= rmst[1]);
            covered_tau += usize::from(tau[0] <= 0.15 && 0.15 <= tau[1]);
        }
        eprintln!(
            "delayed-entry fixed-G survival scalar coverage: RMST {covered_rmst}/2000, S(tau) {covered_tau}/2000"
        );
        assert!(
            (1_860..=1_970).contains(&covered_rmst),
            "delayed-entry fixed-G RMST coverage {covered_rmst}/2000"
        );
        assert!(
            (1_860..=1_970).contains(&covered_tau),
            "delayed-entry fixed-G S(3) coverage {covered_tau}/2000"
        );
    }

    #[test]
    fn delayed_entry_subject_bootstrap_covers_left_truncated_survival_truth() {
        // Entry is independent of the source-population event time. A subject
        // entering at one with an event at one is unobserved; those entering
        // at zero identify the first jump, and later risk sets use both entry
        // cohorts. The two arm laws imply S(3) and RMST differences of .15
        // and .25 respectively.
        let mut state = 0xD3E1_A7E0_95D2_6B63;
        let mut covered_rmst = 0;
        let mut covered_tau = 0;
        for trial in 0..400 {
            let mut durations = Vec::with_capacity(320);
            let mut events = Vec::with_capacity(320);
            let mut treated = Vec::with_capacity(320);
            let mut entries = Vec::with_capacity(320);
            for arm in [false, true] {
                let mut observed = 0;
                while observed < 160 {
                    let entry = (splitmix64(&mut state) & 1) as f64;
                    let event_u = crate::splitmix::splitmix64_unit(&mut state);
                    let (first, second) = if arm { (0.20, 0.15) } else { (0.30, 0.20) };
                    let (exit, event) = if event_u < first {
                        (1.0, 1)
                    } else if event_u < first + second {
                        (2.0, 1)
                    } else {
                        (3.0, 0)
                    };
                    if entry >= exit {
                        continue;
                    }
                    durations.push(exit);
                    events.push(event);
                    treated.push(arm);
                    entries.push(entry);
                    observed += 1;
                }
            }
            let intervals = randomized_survival_bootstrap_intervals(
                &SurvivalRows {
                    duration: &durations,
                    event_code: &events,
                    treated: &treated,
                    delayed_entry: Some(&entries),
                },
                None,
                3.0,
                SurvivalEndpoint::Survival,
                299,
                trial + 2_100,
            )
            .unwrap();
            let rmst = intervals.rmst_difference.unwrap();
            covered_rmst += u32::from(rmst[0] <= 0.25 && 0.25 <= rmst[1]);
            let tau = intervals.difference_at_tau;
            covered_tau += u32::from(tau[0] <= 0.15 && 0.15 <= tau[1]);
        }
        eprintln!(
            "left-truncated survival coverage: RMST {covered_rmst}/400, S(tau) {covered_tau}/400"
        );
        assert!(covered_rmst >= 360, "left-truncated RMST coverage: {covered_rmst}/400");
        assert!(covered_tau >= 360, "left-truncated S(tau) coverage: {covered_tau}/400");
    }

    #[test]
    fn delayed_entry_competing_risk_bootstrap_covers_cif_truth() {
        // Entry at zero or one is independent of event time and cause. People
        // entering at one after a time-one event are absent from the observed
        // sample. Target-cause masses are .15 + .15 in control and .10 + .10
        // in treatment, so the source-population CIF difference at three is
        // -.10; the competing-cause masses are .10 at each event time.
        let mut state = 0xD3E1_C1F0_95D2_6B63;
        let mut covered = 0;
        for trial in 0..400 {
            let mut durations = Vec::with_capacity(320);
            let mut events = Vec::with_capacity(320);
            let mut treated = Vec::with_capacity(320);
            let mut entries = Vec::with_capacity(320);
            for arm in [false, true] {
                let mut observed = 0;
                while observed < 160 {
                    let entry = (splitmix64(&mut state) & 1) as f64;
                    let u = crate::splitmix::splitmix64_unit(&mut state);
                    let first_target = if arm { 0.10 } else { 0.15 };
                    let second_target = if arm { 0.10 } else { 0.15 };
                    let (exit, event) = if u < first_target {
                        (1.0, 1)
                    } else if u < first_target + 0.10 {
                        (1.0, 2)
                    } else if u < first_target + 0.10 + second_target {
                        (2.0, 1)
                    } else if u < first_target + 0.20 + second_target {
                        (2.0, 2)
                    } else {
                        (3.0, 0)
                    };
                    if entry >= exit {
                        continue;
                    }
                    durations.push(exit);
                    events.push(event);
                    treated.push(arm);
                    entries.push(entry);
                    observed += 1;
                }
            }
            let interval = randomized_survival_bootstrap_intervals(
                &SurvivalRows {
                    duration: &durations,
                    event_code: &events,
                    treated: &treated,
                    delayed_entry: Some(&entries),
                },
                None,
                3.0,
                SurvivalEndpoint::CumulativeIncidence { target_cause: 1 },
                299,
                trial + 8_100,
            )
            .unwrap()
            .difference_at_tau;
            covered += u32::from(interval[0] <= -0.10 && -0.10 <= interval[1]);
        }
        eprintln!("left-truncated competing-risk CIF coverage: {covered}/400");
        assert!(covered >= 360, "left-truncated CIF coverage: {covered}/400");
    }

    #[test]
    fn delayed_entry_fixed_g_bootstrap_covers_survival_truth() {
        // Independent entry, independently randomized treatment, and known
        // stratum-specific censoring probabilities. Censoring can occur after
        // time one or two; its survival grid is fixed with each subject.
        let mut state = 0xD3E1_F1E0_95D2_6B63;
        let times = [0.0, 1.0, 1.5, 2.0, 2.5, 3.0];
        let mut covered_rmst = 0;
        let mut covered_tau = 0;
        for trial in 0..400 {
            let mut durations = Vec::with_capacity(400);
            let mut events = Vec::with_capacity(400);
            let mut treated = Vec::with_capacity(400);
            let mut entries = Vec::with_capacity(400);
            let mut g = Vec::with_capacity(400 * times.len());
            for arm in [false, true] {
                let mut observed = 0;
                while observed < 200 {
                    let entry = (splitmix64(&mut state) & 1) as f64;
                    let u = crate::splitmix::splitmix64_unit(&mut state);
                    let (first, second) = if arm { (0.20, 0.15) } else { (0.30, 0.20) };
                    let (event_time, event): (f64, i64) = if u < first {
                        (1.0, 1)
                    } else if u < first + second {
                        (2.0, 1)
                    } else {
                        (3.0, 0)
                    };
                    let high_censoring = splitmix64(&mut state) & 1 == 1;
                    let censor_hazard = if high_censoring { 0.20 } else { 0.10 };
                    let c1 = crate::splitmix::splitmix64_unit(&mut state);
                    let c2 = crate::splitmix::splitmix64_unit(&mut state);
                    let censor_time = if c1 < censor_hazard {
                        1.5
                    } else if c2 < censor_hazard {
                        2.5
                    } else {
                        3.0
                    };
                    let exit = event_time.min(censor_time);
                    if entry >= exit {
                        continue;
                    }
                    durations.push(exit);
                    events.push(if event_time <= censor_time { event } else { 0 });
                    treated.push(arm);
                    entries.push(entry);
                    g.extend_from_slice(&[
                        1.0,
                        1.0,
                        1.0,
                        1.0 - censor_hazard,
                        1.0 - censor_hazard,
                        (1.0 - censor_hazard).powi(2),
                    ]);
                    observed += 1;
                }
            }
            let estimate = randomized_survival_ipcw_summary(
                &SurvivalRows {
                    duration: &durations,
                    event_code: &events,
                    treated: &treated,
                    delayed_entry: Some(&entries),
                },
                &KnownCensoring { times: &times, survival: &g, minimum_probability: 0.01 },
                3.0,
                SurvivalEndpoint::Survival,
            )
            .unwrap()
            .0;
            assert!(estimate.rmst_control.is_some());
            let interval = randomized_survival_bootstrap_intervals(
                &SurvivalRows {
                    duration: &durations,
                    event_code: &events,
                    treated: &treated,
                    delayed_entry: Some(&entries),
                },
                Some(&KnownCensoring { times: &times, survival: &g, minimum_probability: 0.01 }),
                3.0,
                SurvivalEndpoint::Survival,
                299,
                trial + 9_100,
            )
            .unwrap();
            let rmst = interval.rmst_difference.unwrap();
            covered_rmst += u32::from(rmst[0] <= 0.25 && 0.25 <= rmst[1]);
            covered_tau += u32::from(
                interval.difference_at_tau[0] <= 0.15 && 0.15 <= interval.difference_at_tau[1],
            );
        }
        eprintln!(
            "left-entry fixed-G survival coverage: RMST {covered_rmst}/400, S(tau) {covered_tau}/400"
        );
        assert!(covered_rmst >= 360, "left-entry fixed-G RMST coverage: {covered_rmst}/400");
        assert!(covered_tau >= 360, "left-entry fixed-G S(tau) coverage: {covered_tau}/400");
    }

    #[test]
    fn delayed_entry_fixed_g_bootstrap_covers_competing_incidence_truth() {
        let mut state = 0xD3E1_F1CF_95D2_6B63;
        let times = [0.0, 1.0, 1.5, 2.0, 2.5, 3.0];
        let mut covered = 0;
        for trial in 0..400 {
            let mut durations = Vec::with_capacity(400);
            let mut events = Vec::with_capacity(400);
            let mut treated = Vec::with_capacity(400);
            let mut entries = Vec::with_capacity(400);
            let mut g = Vec::with_capacity(400 * times.len());
            for arm in [false, true] {
                let mut observed = 0;
                while observed < 200 {
                    let entry = (splitmix64(&mut state) & 1) as f64;
                    let u = crate::splitmix::splitmix64_unit(&mut state);
                    let target = if arm { 0.10 } else { 0.15 };
                    let (event_time, event): (f64, i64) = if u < target {
                        (1.0, 1)
                    } else if u < target + 0.10 {
                        (1.0, 2)
                    } else if u < 2.0 * target + 0.10 {
                        (2.0, 1)
                    } else if u < 2.0 * target + 0.20 {
                        (2.0, 2)
                    } else {
                        (3.0, 0)
                    };
                    let censor_hazard = if splitmix64(&mut state) & 1 == 1 { 0.20 } else { 0.10 };
                    let c1 = crate::splitmix::splitmix64_unit(&mut state);
                    let c2 = crate::splitmix::splitmix64_unit(&mut state);
                    let censor_time = if c1 < censor_hazard {
                        1.5
                    } else if c2 < censor_hazard {
                        2.5
                    } else {
                        3.0
                    };
                    let exit = event_time.min(censor_time);
                    if entry >= exit {
                        continue;
                    }
                    durations.push(exit);
                    events.push(if event_time <= censor_time { event } else { 0 });
                    treated.push(arm);
                    entries.push(entry);
                    g.extend_from_slice(&[
                        1.0,
                        1.0,
                        1.0,
                        1.0 - censor_hazard,
                        1.0 - censor_hazard,
                        (1.0 - censor_hazard).powi(2),
                    ]);
                    observed += 1;
                }
            }
            let interval = randomized_survival_bootstrap_intervals(
                &SurvivalRows {
                    duration: &durations,
                    event_code: &events,
                    treated: &treated,
                    delayed_entry: Some(&entries),
                },
                Some(&KnownCensoring { times: &times, survival: &g, minimum_probability: 0.01 }),
                3.0,
                SurvivalEndpoint::CumulativeIncidence { target_cause: 1 },
                299,
                trial + 10_100,
            )
            .unwrap()
            .difference_at_tau;
            covered += u32::from(interval[0] <= -0.10 && -0.10 <= interval[1]);
        }
        eprintln!("left-entry fixed-G competing-risk CIF coverage: {covered}/400");
        assert!(covered >= 360, "left-entry fixed-G CIF coverage: {covered}/400");
    }

    #[test]
    fn simultaneous_survival_difference_band_covers_entire_event_grid() {
        let mut state = 0xB41D_95A1_5EED_2026;
        let mut covered = 0;
        for trial in 0..400 {
            let mut durations = Vec::with_capacity(320);
            let mut events = Vec::with_capacity(320);
            let mut treated = Vec::with_capacity(320);
            for arm in [false, true] {
                for _ in 0..160 {
                    let u = crate::splitmix::splitmix64_unit(&mut state);
                    let (first, second) = if arm { (0.15, 0.10) } else { (0.30, 0.20) };
                    let (exit, event) = if u < first {
                        (1.0, 1)
                    } else if u < first + second {
                        (2.0, 1)
                    } else {
                        (3.0, 0)
                    };
                    durations.push(exit);
                    events.push(event);
                    treated.push(arm);
                }
            }
            let band = randomized_survival_bootstrap_difference_band(
                &durations,
                &events,
                &treated,
                3.0,
                SurvivalEndpoint::Survival,
                399,
                trial + 4_400,
            )
            .unwrap();
            let covers = band.times.iter().enumerate().all(|(i, &time)| {
                let truth = if time < 1.0 {
                    0.0
                } else if time < 2.0 {
                    0.15
                } else {
                    0.25
                };
                band.lower[i] <= truth && truth <= band.upper[i]
            });
            covered += u32::from(covers);
        }
        eprintln!("simultaneous survival-grid band coverage: {covered}/400");
        assert!(covered >= 360, "simultaneous survival-grid coverage: {covered}/400");
        assert!(
            randomized_survival_bootstrap_difference_band(
                &[1.0, 3.0, 1.0, 3.0],
                &[1, 0, 1, 0],
                &[false, false, true, true],
                3.0,
                SurvivalEndpoint::Survival,
                399,
                7,
            )
            .unwrap_err()
            .contains("80 subjects")
        );
        let duration = (0..160).map(|i| if i % 5 == 0 { 1.0 } else { 3.0 }).collect::<Vec<_>>();
        let event = duration.iter().map(|time| i64::from(*time == 1.0)).collect::<Vec<_>>();
        let treated = (0..160).map(|i| i >= 80).collect::<Vec<_>>();
        let first = randomized_survival_bootstrap_difference_band(
            &duration,
            &event,
            &treated,
            3.0,
            SurvivalEndpoint::Survival,
            399,
            42,
        )
        .unwrap();
        assert_eq!(
            first,
            randomized_survival_bootstrap_difference_band(
                &duration,
                &event,
                &treated,
                3.0,
                SurvivalEndpoint::Survival,
                399,
                42,
            )
            .unwrap()
        );
    }

    #[test]
    fn simultaneous_competing_incidence_band_covers_entire_event_grid() {
        let mut state = 0xC1F0_95A1_5EED_2026;
        let mut covered = 0;
        for trial in 0..400 {
            let mut durations = Vec::with_capacity(320);
            let mut causes = Vec::with_capacity(320);
            let mut treated = Vec::with_capacity(320);
            for arm in [false, true] {
                for _ in 0..160 {
                    let u = crate::splitmix::splitmix64_unit(&mut state);
                    let (first_target, second_target) =
                        if arm { (0.30, 0.15) } else { (0.20, 0.10) };
                    let (exit, cause) = if u < first_target {
                        (1.0, 1)
                    } else if u < first_target + 0.10 {
                        (1.0, 2)
                    } else if u < first_target + 0.10 + second_target {
                        (2.0, 1)
                    } else if u < first_target + 0.20 + second_target {
                        (2.0, 2)
                    } else {
                        (3.0, 0)
                    };
                    durations.push(exit);
                    causes.push(cause);
                    treated.push(arm);
                }
            }
            let band = randomized_survival_bootstrap_difference_band(
                &durations,
                &causes,
                &treated,
                3.0,
                SurvivalEndpoint::CumulativeIncidence { target_cause: 1 },
                399,
                trial + 6_400,
            )
            .unwrap();
            let covers = band.times.iter().enumerate().all(|(i, &time)| {
                let truth = if time < 1.0 {
                    0.0
                } else if time < 2.0 {
                    0.10
                } else {
                    0.15
                };
                band.lower[i] <= truth && truth <= band.upper[i]
            });
            covered += u32::from(covers);
        }
        eprintln!("simultaneous competing-incidence grid band coverage: {covered}/400");
        assert!(covered >= 360, "simultaneous competing-incidence coverage: {covered}/400");
    }

    #[test]
    fn repeated_sampling_fixed_g_bootstrap_covers_survival_truth() {
        // Known censoring is deliberately heterogeneous and held with the
        // resampled subject. Event risk is randomized and independent of G.
        let mut state = 0xE043_DA75_88C3_9839;
        let mut covered = 0;
        for trial in 0..240 {
            let mut durations = Vec::with_capacity(320);
            let mut events = Vec::with_capacity(320);
            let mut treated = Vec::with_capacity(320);
            let mut g = Vec::with_capacity(320 * 4);
            for i in 0..320 {
                let arm = i >= 160;
                let keep_probability = if i % 2 == 0 { 0.9 } else { 0.7 };
                let censor_u = crate::splitmix::splitmix64_unit(&mut state);
                let event_u = crate::splitmix::splitmix64_unit(&mut state);
                let censored = censor_u >= keep_probability;
                let failed = event_u < if arm { 0.15 } else { 0.30 };
                durations.push(if censored {
                    0.5
                } else if failed {
                    1.0
                } else {
                    3.0
                });
                events.push(i64::from(!censored && failed));
                treated.push(arm);
                g.extend_from_slice(&[1.0, keep_probability, keep_probability, keep_probability]);
            }
            let interval = randomized_survival_bootstrap_intervals(
                &SurvivalRows {
                    duration: &durations,
                    event_code: &events,
                    treated: &treated,
                    delayed_entry: None,
                },
                Some(&KnownCensoring {
                    times: &[0.0, 0.5, 1.0, 3.0],
                    survival: &g,
                    minimum_probability: 0.01,
                }),
                3.0,
                SurvivalEndpoint::Survival,
                299,
                trial + 1700,
            )
            .unwrap()
            .rmst_difference
            .unwrap();
            covered += u32::from(interval[0] <= 0.30 && 0.30 <= interval[1]);
        }
        assert!(covered >= 216, "fixed-G IPCW RMST coverage: {covered}/240");
    }

    #[test]
    fn repeated_sampling_competing_risk_bootstrap_covers_cif_truth() {
        let mut state = 0x8120_D48E_70BA_19D3;
        let mut covered = 0;
        for trial in 0..240 {
            let mut durations = Vec::with_capacity(320);
            let mut events = Vec::with_capacity(320);
            let mut treated = Vec::with_capacity(320);
            let mut g = Vec::with_capacity(320 * 5);
            for i in 0..320 {
                let arm = i >= 160;
                let keep_probability = if i % 2 == 0 { 0.9 } else { 0.7 };
                let censor_u = crate::splitmix::splitmix64_unit(&mut state);
                let cause_u = crate::splitmix::splitmix64_unit(&mut state);
                let censored = censor_u >= keep_probability;
                let first_target = if arm { 0.10 } else { 0.20 };
                let first_competing = first_target + 0.10;
                let later_target = first_competing + 0.10;
                let later_competing = later_target + 0.10;
                let (event_time, cause) = if cause_u < first_target {
                    (1.0, 1)
                } else if cause_u < first_competing {
                    (1.0, 2)
                } else if cause_u < later_target {
                    (2.0, 1)
                } else if cause_u < later_competing {
                    (2.0, 2)
                } else {
                    (3.0, 0)
                };
                durations.push(if censored { 0.5 } else { event_time });
                events.push(if censored { 0 } else { cause });
                treated.push(arm);
                g.extend_from_slice(&[
                    1.0,
                    keep_probability,
                    keep_probability,
                    keep_probability,
                    keep_probability,
                ]);
            }
            let interval = randomized_survival_bootstrap_intervals(
                &SurvivalRows {
                    duration: &durations,
                    event_code: &events,
                    treated: &treated,
                    delayed_entry: None,
                },
                Some(&KnownCensoring {
                    times: &[0.0, 0.5, 1.0, 2.0, 3.0],
                    survival: &g,
                    minimum_probability: 0.01,
                }),
                3.0,
                SurvivalEndpoint::CumulativeIncidence { target_cause: 1 },
                299,
                trial + 2700,
            )
            .unwrap()
            .difference_at_tau;
            covered += u32::from(interval[0] <= -0.10 && -0.10 <= interval[1]);
        }
        assert!(covered >= 216, "fixed-G competing-risk CIF coverage: {covered}/240");
    }

    #[test]
    fn known_censoring_left_limit_handles_event_censor_ties() {
        let g = [
            1.0, 0.5, 0.25, 1.0, 0.5, 0.25, 1.0, 1.0, 0.5, 1.0, 1.0, 0.5, 1.0, 1.0, 0.5, 1.0, 1.0,
            0.5,
        ];
        let (summary, _) = randomized_survival_ipcw_summary(
            &SurvivalRows {
                duration: &[1.0, 1.0, 2.0, 1.0, 1.0, 2.0],
                event_code: &[1, 0, 0, 1, 0, 0],
                treated: &[false, false, false, true, true, true],
                delayed_entry: None,
            },
            &KnownCensoring { times: &[0.0, 1.0, 2.0], survival: &g, minimum_probability: 0.1 },
            2.0,
            SurvivalEndpoint::Survival,
        )
        .unwrap();
        // At time 1 the tied censor remains at risk: control weighted event 2,
        // risk 2+2+1=5; treated event 1 and risk 3.
        assert!((summary.control[1] - 0.6).abs() < 1e-12);
        assert!((summary.treated[1] - 2.0 / 3.0).abs() < 1e-12);
        assert!((summary.rmst_control.unwrap() - 1.6).abs() < 1e-12);
    }

    #[test]
    fn known_censoring_ipcw_survival_matches_unweighted_truth_when_g_is_one() {
        let (summary, minimum_g) = randomized_survival_ipcw_summary(
            &SurvivalRows {
                duration: &[1.0, 2.0, 2.0, 2.0],
                event_code: &[1, 0, 0, 0],
                treated: &[false, false, true, true],
                delayed_entry: None,
            },
            &KnownCensoring {
                times: &[0.0, 1.0, 2.0],
                survival: &[1.0; 12],
                minimum_probability: 0.1,
            },
            2.0,
            SurvivalEndpoint::Survival,
        )
        .unwrap();
        assert_eq!(summary.control, [1.0, 0.5, 0.5]);
        assert_eq!(summary.treated, [1.0, 1.0, 1.0]);
        assert_eq!(summary.rmst_control, Some(1.5));
        assert_eq!(summary.rmst_treated, Some(2.0));
        assert_eq!(minimum_g, 1.0);
    }

    #[test]
    fn known_censoring_ipcw_refuses_positivity_failure() {
        let mut g = [1.0; 12];
        g[4] = 0.05;
        assert!(
            randomized_survival_ipcw_summary(
                &SurvivalRows {
                    duration: &[1.0, 2.0, 2.0, 2.0],
                    event_code: &[1, 0, 0, 0],
                    treated: &[false, false, true, true],
                    delayed_entry: None
                },
                &KnownCensoring { times: &[0.0, 1.0, 2.0], survival: &g, minimum_probability: 0.1 },
                2.0,
                SurvivalEndpoint::Survival
            )
            .is_err()
        );
    }

    #[test]
    fn known_censoring_ipcw_competing_risk_known_truth() {
        let (summary, _) = randomized_survival_ipcw_summary(
            &SurvivalRows {
                duration: &[1.0, 2.0, 2.0, 1.0, 2.0, 2.0],
                event_code: &[1, 2, 0, 1, 2, 0],
                treated: &[false, false, false, true, true, true],
                delayed_entry: None,
            },
            &KnownCensoring {
                times: &[0.0, 1.0, 2.0],
                survival: &[1.0; 18],
                minimum_probability: 0.1,
            },
            2.0,
            SurvivalEndpoint::CumulativeIncidence { target_cause: 1 },
        )
        .unwrap();
        assert_eq!(summary.control, [0.0, 1.0 / 3.0, 1.0 / 3.0]);
        assert_eq!(summary.treated, [0.0, 1.0 / 3.0, 1.0 / 3.0]);
        assert_eq!(summary.rmst_control, None);
        assert_eq!(summary.rmst_treated, None);
    }

    #[test]
    fn randomized_rmst_and_survival_known_truth() {
        let result = randomized_survival_summary(
            &SurvivalRows {
                duration: &[1.0, 2.0, 2.0, 2.0],
                event_code: &[1, 0, 0, 0],
                treated: &[false, false, true, true],
                delayed_entry: None,
            },
            2.0,
            SurvivalEndpoint::Survival,
        )
        .unwrap();
        assert_eq!(result.times, [0.0, 1.0, 2.0]);
        assert_eq!(result.control, [1.0, 0.5, 0.5]);
        assert_eq!(result.treated, [1.0, 1.0, 1.0]);
        assert_eq!(result.rmst_control, Some(1.5));
        assert_eq!(result.rmst_treated, Some(2.0));
    }

    #[test]
    fn delayed_entry_risk_sets_and_competing_incidence_known_truth() {
        let survival = randomized_survival_summary(
            &SurvivalRows {
                duration: &[1.0, 2.0, 2.0, 2.0],
                event_code: &[1, 0, 0, 0],
                treated: &[false, false, true, true],
                delayed_entry: Some(&[0.0, 0.5, 0.0, 0.5]),
            },
            2.0,
            SurvivalEndpoint::Survival,
        )
        .unwrap();
        assert_eq!(survival.rmst_control, Some(1.5));
        assert_eq!(survival.minimum_event_risk_set, Some(2));

        let incidence = randomized_survival_summary(
            &SurvivalRows {
                duration: &[1.0, 2.0, 1.0, 2.0],
                event_code: &[1, 2, 1, 1],
                treated: &[false, false, true, true],
                delayed_entry: None,
            },
            2.0,
            SurvivalEndpoint::CumulativeIncidence { target_cause: 1 },
        )
        .unwrap();
        assert_eq!(incidence.control, [0.0, 0.5, 0.5]);
        assert_eq!(incidence.treated, [0.0, 0.5, 1.0]);
        assert_eq!(incidence.rmst_control, None);
    }

    #[test]
    fn refuses_unsupported_risk_sets_and_codes() {
        assert!(
            randomized_survival_summary(
                &SurvivalRows {
                    duration: &[1.0, 2.0],
                    event_code: &[1, 2],
                    treated: &[false, true],
                    delayed_entry: None
                },
                2.0,
                SurvivalEndpoint::Survival
            )
            .is_err()
        );
        assert!(
            randomized_survival_summary(
                &SurvivalRows {
                    duration: &[1.0, 2.0],
                    event_code: &[1, 0],
                    treated: &[false, true],
                    delayed_entry: Some(&[0.0, 2.0])
                },
                2.0,
                SurvivalEndpoint::Survival
            )
            .is_err()
        );
    }
}
