//! The facade group-time ATT and the compiled staggered panel `DiD` routes give identical
//! numbers.
//!
//! `antecedent::estimate::staggered_event_study::group_time_att` is reproduced by two compiled
//! queries: the selected group-time query, one cohort-period cell per query, and the staggered
//! event study, whose post-adoption event-time effects are the same cells. The selected
//! group-time route calls `group_time_att` itself on the balanced panels it accepts whose
//! selected cell keeps treated and never-treated clusters apart, as every cell here does; the
//! event study runs the event-time kernel, which forms the same contrasts. The assertions are
//! on `to_bits()` of the effect and standard error; a calibrated interval exists only on the
//! event-study route, so its bounds are checked against the kernel interval function applied
//! to the direct values.
#![allow(
    clippy::float_cmp,
    clippy::cast_lossless,
    reason = "bit-identical equality is the property under test; fixtures widen small integers"
)]

use std::sync::Arc;

use antecedent::estimate::staggered_event_study::group_time_att;
use antecedent::prelude::ExecutionContext;
use antecedent::{PanelDidQuery, Study};
use antecedent_core::VariableId;
use antecedent_data::TabularData;
use antecedent_estimate::staggered_event_study::{
    EventTimeEffect, GroupTimeAtt, pointwise_interval_95,
};

/// A balanced staggered panel with cohorts 0 (never treated), 3 and 4 over periods 1 to 5.
struct Panel {
    outcome: Vec<f64>,
    subjects: Vec<Arc<str>>,
    clusters: Vec<Arc<str>>,
    periods: Vec<i64>,
    cohorts: Vec<i64>,
}

fn panel(subjects_per_cohort: usize) -> Panel {
    let mut panel = Panel {
        outcome: Vec::new(),
        subjects: Vec::new(),
        clusters: Vec::new(),
        periods: Vec::new(),
        cohorts: Vec::new(),
    };
    for (cohort_index, cohort) in [0_i64, 3, 4].into_iter().enumerate() {
        for unit in 0..subjects_per_cohort {
            let id = Arc::<str>::from(format!("u{cohort_index}-{unit}"));
            // Two subjects share a cluster, so the cluster count differs from the subject count.
            let cluster = Arc::<str>::from(format!("k{cohort_index}-{}", unit / 2));
            for period in 1..=5_i64 {
                let wobble = (((unit * 7 + period as usize * 13 + cohort_index * 5) % 11) as f64
                    - 5.0)
                    * 0.173;
                let effect = if cohort > 0 && period >= cohort {
                    1.5 + 0.25 * (period - cohort) as f64
                } else {
                    0.0
                };
                panel.outcome.push(2.0 + 0.4 * unit as f64 + 1.1 * period as f64 + wobble + effect);
                panel.subjects.push(id.clone());
                panel.clusters.push(cluster.clone());
                panel.periods.push(period);
                panel.cohorts.push(cohort);
            }
        }
    }
    panel
}

fn direct(panel: &Panel) -> Vec<GroupTimeAtt> {
    group_time_att(&panel.outcome, &panel.subjects, &panel.periods, &panel.cohorts, &panel.clusters)
        .unwrap()
}

fn data(panel: &Panel) -> TabularData {
    TabularData::from_f64_columns([("outcome", panel.outcome.as_slice())]).unwrap()
}

#[test]
fn selected_group_time_query_reproduces_every_direct_cell() {
    let panel = panel(8);
    let data = data(&panel);
    let context = ExecutionContext::for_tests(7);
    let cells = direct(&panel);
    assert!(cells.len() >= 5, "the fixture must exercise several cohort-period cells");
    for cell in &cells {
        let query = PanelDidQuery::staggered_group_time(
            VariableId::from_raw(0),
            panel.subjects.clone(),
            panel.clusters.clone(),
            panel.periods.clone(),
            panel.cohorts.clone(),
            cell.cohort,
            cell.period,
        );
        let result =
            Study::tabular(data.clone()).query(query).build().unwrap().run(&context).unwrap();
        let compiled = result.panel_did().unwrap();
        let at = format!("cohort {} period {}", cell.cohort, cell.period);
        assert_eq!(compiled.effect.to_bits(), cell.effect.to_bits(), "{at}: effect");
        assert_eq!(
            compiled.standard_error.to_bits(),
            cell.standard_error.to_bits(),
            "{at}: standard error"
        );
        assert_eq!(compiled.treated_subjects, cell.treated_subjects, "{at}");
        assert_eq!(compiled.comparison_subjects, cell.comparison_subjects, "{at}");
        assert_eq!(compiled.clusters, cell.clusters, "{at}");
        // Neither path publishes an interval for a selected cell.
        assert!(compiled.interval_95.is_none(), "{at}");
    }
}

#[test]
fn event_study_post_adoption_effects_reproduce_the_direct_group_time_cells() {
    let panel = panel(8);
    let data = data(&panel);
    let context = ExecutionContext::for_tests(7);
    let cells = direct(&panel);
    let query = PanelDidQuery::staggered_event_study(
        VariableId::from_raw(0),
        panel.subjects.clone(),
        panel.clusters.clone(),
        panel.periods.clone(),
        panel.cohorts.clone(),
    );
    let result = Study::tabular(data).query(query).build().unwrap().run(&context).unwrap();
    let curve = &result.panel_did().unwrap().event_time_effects;
    let post: Vec<&EventTimeEffect> =
        curve.iter().filter(|effect| effect.event_time >= 0).collect();
    assert_eq!(post.len(), cells.len());
    for (compiled, cell) in post.iter().zip(&cells) {
        assert_eq!((compiled.cohort, compiled.period), (cell.cohort, cell.period));
        assert_eq!(compiled.effect.to_bits(), cell.effect.to_bits());
        assert_eq!(compiled.standard_error.to_bits(), cell.standard_error.to_bits());
        assert_eq!(compiled.treated_subjects, cell.treated_subjects);
        assert_eq!(compiled.comparison_subjects, cell.comparison_subjects);
        assert_eq!(compiled.clusters, cell.clusters);
    }
}

#[test]
fn event_study_interval_bounds_are_the_kernel_interval_of_the_direct_values() {
    // Enough clusters on each side for the pointwise interval to be published.
    let panel = panel(60);
    let data = data(&panel);
    let context = ExecutionContext::for_tests(7);
    let cells = direct(&panel);
    let query = PanelDidQuery::staggered_event_study(
        VariableId::from_raw(0),
        panel.subjects.clone(),
        panel.clusters.clone(),
        panel.periods.clone(),
        panel.cohorts.clone(),
    );
    let result = Study::tabular(data).query(query).build().unwrap().run(&context).unwrap();
    let estimate = result.panel_did().unwrap();
    let treated_clusters = |cohort: i64| {
        let mut names: Vec<&Arc<str>> = panel
            .clusters
            .iter()
            .zip(&panel.cohorts)
            .filter(|(_, c)| **c == cohort)
            .map(|(cluster, _)| cluster)
            .collect();
        names.sort();
        names.dedup();
        names.len()
    };
    let control_clusters = treated_clusters(0);
    let mut published = 0;
    for (index, effect) in estimate.event_time_effects.iter().enumerate() {
        if effect.event_time < 0 {
            assert!(estimate.event_time_intervals_95[index].is_none());
            continue;
        }
        let cell = cells
            .iter()
            .find(|cell| (cell.cohort, cell.period) == (effect.cohort, effect.period))
            .unwrap();
        let direct_effect = EventTimeEffect {
            cohort: cell.cohort,
            period: cell.period,
            event_time: cell.period - cell.cohort,
            effect: cell.effect,
            treated_subjects: cell.treated_subjects,
            comparison_subjects: cell.comparison_subjects,
            standard_error: cell.standard_error,
            clusters: cell.clusters,
        };
        let expected =
            pointwise_interval_95(&direct_effect, treated_clusters(cell.cohort), control_clusters);
        let compiled = estimate.event_time_intervals_95[index];
        assert_eq!(compiled.is_some(), expected.is_some());
        if let (Some(compiled), Some(expected)) = (compiled, expected) {
            assert_eq!(compiled[0].to_bits(), expected[0].to_bits());
            assert_eq!(compiled[1].to_bits(), expected[1].to_bits());
            published += 1;
        }
    }
    assert!(published > 0, "the fixture must publish at least one interval");
}
