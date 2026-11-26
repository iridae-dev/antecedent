//! Compiled fixed schedule comparison and honestly learned longitudinal policies.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::Study;
use antecedent::analysis::composition::{BundleBuilder, consume, export_bundle};
use antecedent_core::{
    CausalQuery, ExecutionContext, LongitudinalRegimeMethod, LongitudinalRegimeQuery,
    LongitudinalScheduleFamily, LongitudinalScheduleTarget, StreamDomain, VariableId,
};
use antecedent_data::TabularData;

const SUBJECTS: usize = 800;
const DRAWS: u32 = 199;

fn fixture(
    means: [f64; 4],
    target: LongitudinalScheduleTarget,
) -> (TabularData, LongitudinalRegimeQuery, Vec<f64>) {
    let outcomes: Vec<_> =
        (0..SUBJECTS).map(|row| means[row % 4] + 0.125 * ((row / 8 % 3) as f64 - 1.0)).collect();
    let q = LongitudinalRegimeQuery {
        outcome: VariableId::from_raw(0),
        method: LongitudinalRegimeMethod::ScheduleFamily(LongitudinalScheduleFamily {
            schedules: 0b1111,
            target,
            draws: DRAWS,
        }),
        period_outcome_predictions: None,
        known_fixed_outcome_predictions: false,
        stabilizing_numerator_probabilities: None,
        q_predictions: None,
        observation_history: None,
        prediction_fold_ids: None,
        periods: 2,
        treatment_history: (0..SUBJECTS).flat_map(|row| [row & 2 != 0, row & 1 != 0]).collect(),
        regime_actions: vec![false; SUBJECTS * 2].into(),
        treatment_probabilities: vec![0.5; SUBJECTS * 2].into(),
        censoring_probabilities: vec![1.0; SUBJECTS * 2].into(),
        outcome_observed: vec![true; SUBJECTS].into(),
        subject_ids: (0..SUBJECTS).map(|i| Arc::from(format!("schedule-{i}"))).collect(),
        fold_ids: (0..SUBJECTS).map(|row| u32::try_from((row / 4) % 2).unwrap()).collect(),
        excluded_fold_predictions: false,
        probabilities_known_by_design: true,
        minimum_probability: 0.01,
        rule_id: None,
        rule_version: None,
        rule_provenance: None,
    };
    let table = TabularData::from_f64_columns([("outcome", outcomes.as_slice())]).unwrap();
    (table, q, outcomes)
}

fn compiled(
    table: &TabularData,
    q: LongitudinalRegimeQuery,
    ctx: &ExecutionContext,
) -> antecedent::PreparedStudy {
    Study::tabular(table.clone())
        .query(CausalQuery::LongitudinalRegime(q))
        .build()
        .unwrap()
        .prepare(ctx)
        .unwrap()
}

#[test]
fn simultaneous_schedule_intervals_use_paired_subject_scores_and_bound_the_contrast_family() {
    let target = LongitudinalScheduleTarget::Contrasts { reference: 0, contrasts: 0b1010 };
    let (table, mut q, outcomes) = fixture([1.0, 2.0, 3.0, 10.0], target);
    let ctx = ExecutionContext::for_tests(408);
    let first = compiled(&table, q.clone(), &ctx);
    let result = first.estimate(&table, &ctx).unwrap();
    let family = result.longitudinal_regime().unwrap().schedule_family.as_ref().unwrap();
    let contrasts: [Vec<f64>; 2] = [1, 3].map(|candidate| {
        outcomes
            .iter()
            .enumerate()
            .map(|(i, &y)| 4.0 * y * (f64::from(i % 4 == candidate) - f64::from(i % 4 == 0)))
            .collect()
    });
    let means = contrasts.each_ref().map(|rows| rows.iter().sum::<f64>() / SUBJECTS as f64);
    let norms =
        [0, 1].map(|i| contrasts[i].iter().map(|row| (row - means[i]).powi(2)).sum::<f64>().sqrt());
    let mut maxima: Vec<_> = (0..DRAWS)
        .map(|draw| {
            let mut rng =
                ctx.rng.stream_for(StreamDomain::Resample, 0x5319_0000_0000_0000 | u64::from(draw));
            let noise: Vec<_> =
                (0..SUBJECTS).map(|_| antecedent_kernels::standard_normal(&mut rng)).collect();
            let stats = [0, 1].map(|candidate| {
                let centered: f64 = contrasts[candidate]
                    .iter()
                    .zip(&noise)
                    .map(|(&score, &multiplier)| (score - means[candidate]) * multiplier)
                    .sum();
                (centered / norms[candidate]).abs()
            });
            stats[0].max(stats[1])
        })
        .collect();
    maxima.sort_by(f64::total_cmp);
    let critical = maxima[189]; // the 190th of 199 at one-based rank 0.95 × 200
    assert!((critical - family.critical_value).abs() < 1e-12);
    for (i, contrast) in family.contrasts.iter().enumerate() {
        let se = norms[i] / ((SUBJECTS * (SUBJECTS - 1)) as f64).sqrt();
        assert!((contrast.difference - means[i]).abs() < 1e-12);
        assert!((contrast.interval[0] - (means[i] - critical * se)).abs() < 1e-12);
        assert!((contrast.interval[1] - (means[i] + critical * se)).abs() < 1e-12);
    }
    q.method = LongitudinalRegimeMethod::ScheduleFamily(LongitudinalScheduleFamily {
        schedules: 0b1011,
        target,
        draws: DRAWS,
    });
    let revised = compiled(&table, q, &ctx);
    assert_ne!(first.plan_view().unwrap().identity(), revised.plan_view().unwrap().identity());
    let sourced = first
        .encode_contracted_result_with_source(&result, &table, "fixed-schedules", &ctx)
        .unwrap();
    let mut builder = BundleBuilder::new();
    builder.add_detected_artifact(Some("comparison"), &sourced).unwrap();
    let bundle = builder.build().unwrap();
    let bytes = export_bundle(&bundle, "fixed-schedule-bundle").unwrap();
    let replayed = consume(&bytes, bundle.identity(), &Default::default()).unwrap();
    assert!(replayed.all_verified());
    assert_eq!(
        replayed.value("comparison", "schedule.3.versus.0.upper_95"),
        Some(family.contrasts[1].interval[1])
    );
}

