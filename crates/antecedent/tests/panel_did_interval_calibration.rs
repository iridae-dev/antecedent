//! Repeated-sampling evidence for retained two-period `DiD` cluster scores.
//! SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(
    clippy::cast_lossless,
    clippy::cast_sign_loss,
    reason = "fixture derives f64 baselines and a u64 seed offset from small nonnegative integer cluster, subject, and row counters"
)]

use std::sync::Arc;

use antecedent::prelude::ExecutionContext;
use antecedent::{PanelDidQuery, Study};
use antecedent_core::{CausalRng, VariableId};
use antecedent_data::TabularData;
use antecedent_kernels::standard_normal;

const REPLICATIONS: usize = 2_000;
const TRUTH: f64 = 2.0;
const CRITICAL_95: f64 = 1.959_963_984_540_054;

fn panel_fixture(clusters_per_arm: usize) -> (PanelDidQuery, Vec<f64>, Vec<bool>) {
    let mut subjects = Vec::new();
    let mut clusters = Vec::new();
    let mut treated = Vec::new();
    let mut post = Vec::new();
    let mut baseline = Vec::new();
    for cluster in 0..2 * clusters_per_arm {
        let active = cluster >= clusters_per_arm;
        for subject in 0..2 {
            for after in [false, true] {
                subjects.push(Arc::<str>::from(format!("subject-{cluster}-{subject}")));
                clusters.push(Arc::<str>::from(format!("cluster-{cluster}")));
                treated.push(active);
                post.push(after);
                baseline.push(cluster as f64 * 0.01 + subject as f64 * 0.1);
            }
        }
    }
    (
        PanelDidQuery::new(VariableId::from_raw(0), treated.clone(), post, subjects, clusters),
        baseline,
        treated,
    )
}

#[test]
fn retained_panel_did_cluster_normal_interval_covers_parallel_trends_truth() {
    for clusters_per_arm in [30, 80] {
        let (query, baseline, treated) = panel_fixture(clusters_per_arm);
        let data = TabularData::from_f64_columns([("outcome", baseline.as_slice())]).unwrap();
        let ctx = ExecutionContext::for_tests(20_260_927);
        let prepared = Study::tabular(data).query(query).build().unwrap().prepare(&ctx).unwrap();
        let mut rng = CausalRng::from_seed(0x21_09_27_03 + clusters_per_arm as u64);
        let mut covered = 0;
        for _ in 0..REPLICATIONS {
            let mut outcome = baseline.clone();
            for cluster in 0..2 * clusters_per_arm {
                let cluster_shock = standard_normal(&mut rng);
                for subject in 0..2 {
                    let row = (cluster * 2 + subject) * 2;
                    outcome[row + 1] += 0.5
                        + if treated[row] { TRUTH } else { 0.0 }
                        + cluster_shock
                        + 0.5 * standard_normal(&mut rng);
                }
            }
            let data = TabularData::from_f64_columns([("outcome", outcome.as_slice())]).unwrap();
            let result = prepared.estimate(&data, &ctx).unwrap();
            let fit = result.panel_did.unwrap();
            assert_eq!(
                fit.uncertainty.as_ref(),
                "cluster_robust_normal_interval_independent_clusters"
            );
            assert!(fit.interval_95.is_some());
            covered += usize::from((fit.effect - TRUTH).abs() <= CRITICAL_95 * fit.standard_error);
        }
        let rate = covered as f64 / REPLICATIONS as f64;
        eprintln!(
            "panel DiD {clusters_per_arm} clusters per arm: {covered}/{REPLICATIONS} = {rate:.4}"
        );
        assert!((0.925..=0.975).contains(&rate), "panel DiD coverage {rate:.4}");
    }
}

#[test]
fn retained_staggered_event_study_post_adoption_interval_covers_parallel_trends_truth() {
    // Never-treated controls (cohort 0) versus a single adoption cohort at
    // period three; one subject per cluster so the CR1 score sandwich has one
    // independent cluster per subject. The representative post-adoption
    // interval (event time zero) targets the TRUTH treatment effect.
    let clusters_per_group = 24;
    let mut subjects = Vec::new();
    let mut clusters = Vec::new();
    let mut periods = Vec::new();
    let mut cohorts = Vec::new();
    for (group, cohort) in [("control", 0_i64), ("treated", 3)] {
        for cluster in 0..clusters_per_group {
            let id = Arc::<str>::from(format!("{group}-{cluster}"));
            for period in 1..=4_i64 {
                subjects.push(id.clone());
                clusters.push(id.clone());
                periods.push(period);
                cohorts.push(cohort);
            }
        }
    }
    let zeros = vec![0.0; subjects.len()];
    let query = PanelDidQuery::staggered_event_study(
        VariableId::from_raw(0),
        subjects,
        clusters.clone(),
        periods.clone(),
        cohorts.clone(),
    );
    let ctx = ExecutionContext::for_tests(20_260_929);
    let prepared =
        Study::tabular(TabularData::from_f64_columns([("outcome", zeros.as_slice())]).unwrap())
            .query(query)
            .build()
            .unwrap()
            .prepare(&ctx)
            .unwrap();
    let mut rng = CausalRng::from_seed(0x21_09_27_05);
    let mut covered = 0;
    for _ in 0..REPLICATIONS {
        let outcome: Vec<f64> = periods
            .iter()
            .zip(&cohorts)
            .map(|(&period, &cohort)| {
                0.5 * period as f64
                    + if cohort == 3 && period >= 3 { TRUTH } else { 0.0 }
                    + standard_normal(&mut rng)
            })
            .collect();
        let data = TabularData::from_f64_columns([("outcome", outcome.as_slice())]).unwrap();
        let result = prepared.estimate(&data, &ctx).unwrap();
        let fit = result.panel_did.unwrap();
        assert_eq!(
            fit.uncertainty.as_ref(),
            "event_time_pointwise_normal_intervals_independent_clusters"
        );
        let interval_95 =
            fit.interval_95.expect("supported post-adoption interval at 24 clusters per group");
        covered += usize::from(interval_95[0] <= TRUTH && TRUTH <= interval_95[1]);
    }
    let rate = covered as f64 / REPLICATIONS as f64;
    eprintln!(
        "staggered event study {clusters_per_group} clusters per group: {covered}/{REPLICATIONS} = {rate:.4}"
    );
    assert!((0.925..=0.975).contains(&rate), "staggered event-study coverage {rate:.4}");
}

#[test]
fn retained_repeated_cross_section_did_interval_covers_parallel_trends_truth() {
    for rows_per_cell in [30, 100] {
        let mut treated = Vec::new();
        let mut post = Vec::new();
        let mut subjects = Vec::new();
        let mut clusters = Vec::new();
        for group in [false, true] {
            for after in [false, true] {
                for row in 0..rows_per_cell {
                    treated.push(group);
                    post.push(after);
                    subjects.push(Arc::<str>::from(format!("subject-{group}-{after}-{row}")));
                    clusters.push(Arc::<str>::from(format!("cluster-{group}-{after}-{row}")));
                }
            }
        }
        let query = PanelDidQuery::repeated_cross_section(
            VariableId::from_raw(0),
            treated.clone(),
            post.clone(),
            subjects,
            clusters,
        );
        let zeros = vec![0.0; treated.len()];
        let ctx = ExecutionContext::for_tests(20_260_928);
        let prepared =
            Study::tabular(TabularData::from_f64_columns([("outcome", zeros.as_slice())]).unwrap())
                .query(query)
                .build()
                .unwrap()
                .prepare(&ctx)
                .unwrap();
        let mut rng = CausalRng::from_seed(0x21_09_27_04 + rows_per_cell as u64);
        let mut covered = 0;
        for _ in 0..REPLICATIONS {
            let outcome: Vec<f64> = treated
                .iter()
                .zip(&post)
                .map(|(&active, &after)| {
                    f64::from(active)
                        + 0.5 * f64::from(after)
                        + TRUTH * f64::from(active && after)
                        + standard_normal(&mut rng)
                })
                .collect();
            let data = TabularData::from_f64_columns([("outcome", outcome.as_slice())]).unwrap();
            let result = prepared.estimate(&data, &ctx).unwrap();
            let fit = result.panel_did.unwrap();
            assert_eq!(
                fit.uncertainty.as_ref(),
                "cluster_robust_normal_interval_independent_clusters"
            );
            assert!(fit.interval_95.is_some());
            covered += usize::from((fit.effect - TRUTH).abs() <= CRITICAL_95 * fit.standard_error);
        }
        let rate = covered as f64 / REPLICATIONS as f64;
        eprintln!(
            "repeated cross-section DiD {rows_per_cell} rows per cell: {covered}/{REPLICATIONS} = {rate:.4}"
        );
        assert!((0.925..=0.975).contains(&rate), "repeated cross-section DiD coverage {rate:.4}");
    }
}
