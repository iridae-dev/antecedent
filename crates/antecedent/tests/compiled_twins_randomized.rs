//! Twin equality of the randomized-design direct kernels and their compiled routes.
//!
//! Each test runs a direct facade kernel and the `RandomizedEffect` study that replaces it on
//! the same rows, and asserts bit-identical point estimate, standard error and interval bounds.
//! The compiled routes call the same kernel with the same inputs, so any difference is a bug.
//! The probability-weighted exact test, same-sample CUPED, ANCOVA, stratified and switchback
//! doors are bit twins; the independent-Bernoulli 2x2 factorial door is not (the compiled
//! route is the four-cell multi-arm route, which sums in a different order), and its test
//! states the tolerance it holds the two to.
//!
//! Seeds: none of these kernels draws randomness, so the direct path and the compiled route
//! share no seed to pin; the compiled route's context seed (`ExecutionContext::for_tests`) does
//! not reach them.
// SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(deprecated, reason = "the direct facade doors are the paths under comparison")]

use std::sync::Arc;

use antecedent::prelude::ExecutionContext;
use antecedent::{RandomizedEffectEstimate, RandomizedEffectQuery, Study};
use antecedent_core::{RandomizationDesign, VariableId};
use antecedent_data::TabularData;

fn labels(prefix: &str, n: usize) -> Vec<Arc<str>> {
    (0..n).map(|i| Arc::<str>::from(format!("{prefix}-{i}"))).collect()
}

fn bernoulli(n: usize, assigned: Vec<bool>, probabilities: Vec<f64>) -> RandomizedEffectQuery {
    RandomizedEffectQuery::bernoulli_itt(
        VariableId::from_raw(0),
        assigned,
        probabilities,
        labels("unit", n),
        labels("row", n),
        ("control", "treated"),
    )
}

fn run(data: &TabularData, query: RandomizedEffectQuery) -> RandomizedEffectEstimate {
    let ctx = ExecutionContext::for_tests(0x7717_0A);
    Study::tabular(data.clone())
        .query(query)
        .build()
        .unwrap()
        .run(&ctx)
        .unwrap()
        .into_randomized_effect()
        .unwrap()
}

fn same_interval(left: Option<[f64; 2]>, right: Option<[f64; 2]>) {
    match (left, right) {
        (Some(a), Some(b)) => {
            assert_eq!(a[0].to_bits(), b[0].to_bits(), "lower bound");
            assert_eq!(a[1].to_bits(), b[1].to_bits(), "upper bound");
        }
        (None, None) => {}
        other => panic!("interval presence differs: {other:?}"),
    }
}

/// Same-sample CUPED: the compiled route is `estimated_cuped_contrast`, which wraps
/// `cuped_ht_effect`; the effect, the pooled slope's standard error and the interval agree.
#[test]
fn estimated_slope_cuped_matches_the_direct_kernel() {
    let n = 400;
    let x = (0..n).map(|i| (0.17 * i as f64).sin() + 0.01 * i as f64).collect::<Vec<_>>();
    let assigned = (0..n).map(|i| (i * 7 + i / 5) % 2 == 0).collect::<Vec<_>>();
    let y = (0..n)
        .map(|i| 2.0 + 1.3 * x[i] + 0.4 * (0.29 * i as f64).sin() + 1.4 * f64::from(assigned[i]))
        .collect::<Vec<_>>();
    let data =
        TabularData::from_f64_columns([("outcome", y.as_slice()), ("baseline", x.as_slice())])
            .unwrap();
    let probabilities = vec![0.5; n];
    let direct =
        antecedent::estimate::randomized_scores::cuped_ht_effect(&y, &x, &assigned, &probabilities)
            .unwrap();
    let query = bernoulli(n, assigned.clone(), probabilities.clone())
        .with_estimated_cuped(VariableId::from_raw(1));
    assert_eq!(
        query.graphless_coordinate(),
        (
            "randomized_effect",
            "bernoulli",
            "estimated_cuped_ht_score",
            "unit",
            "pointwise_95_normal_interval"
        )
    );
    let fit = run(&data, query);
    assert_eq!(fit.effect.to_bits(), direct.effect.to_bits());
    assert_eq!(
        fit.variance_upper_bound.to_bits(),
        (direct.standard_error * direct.standard_error).to_bits()
    );
    let standard_error = fit.standard_error.expect("a supported design publishes its interval");
    assert_eq!(standard_error.to_bits(), direct.standard_error.to_bits());
    let reference = antecedent_estimate::randomized_scores::estimated_cuped_contrast(
        &y,
        &x,
        &assigned,
        &probabilities,
    )
    .unwrap();
    assert_eq!(reference.beta.to_bits(), direct.beta.to_bits());
    assert!(reference.interval_95.is_some());
    same_interval(fit.interval_95, reference.interval_95);
    assert_eq!(fit.treatment_units, assigned.iter().filter(|a| **a).count());
    assert_eq!(fit.control_units, n - fit.treatment_units);
}

/// The same-sample slope is not the fixed route: a coefficient fixed at the slope the kernel
/// estimates reproduces a different contrast (the fixed route does not centre the covariate), so
/// the two routes are kept apart by their query declarations.
#[test]
fn estimated_and_fixed_cuped_are_distinct_declarations() {
    let n = 400;
    let x = (0..n).map(|i| (0.17 * i as f64).sin() + 0.01 * i as f64).collect::<Vec<_>>();
    let assigned = (0..n).map(|i| i % 2 == 0).collect::<Vec<_>>();
    let base = bernoulli(n, assigned, vec![0.5; n]);
    let estimated = base.clone().with_estimated_cuped(VariableId::from_raw(1));
    let fixed = base.with_fixed_cuped(VariableId::from_raw(1), 1.3);
    assert_ne!(estimated.graphless_coordinate().2, fixed.graphless_coordinate().2);
    assert!(estimated.clone().with_fixed_cuped(VariableId::from_raw(1), 1.3).validate().is_err());
    assert!(estimated.with_ancova(vec![VariableId::from_raw(2)]).validate().is_err());
}

/// ANCOVA: the compiled route is `fit_ancova` plus the calibrated Bernoulli interval.
#[test]
fn ancova_matches_the_direct_kernel() {
    let n = 400;
    let x1 = (0..n).map(|i| (0.17 * i as f64).sin()).collect::<Vec<_>>();
    let x2 = (0..n).map(|i| (0.11 * i as f64).cos()).collect::<Vec<_>>();
    let assigned = (0..n).map(|i| i % 2 == 0).collect::<Vec<_>>();
    let y = (0..n)
        .map(|i| {
            2.0 + 1.2 * x1[i] - 0.7 * x2[i]
                + 0.5 * (0.29 * i as f64).sin()
                + 1.4 * f64::from(assigned[i])
        })
        .collect::<Vec<_>>();
    let data = TabularData::from_f64_columns([
        ("outcome", y.as_slice()),
        ("baseline_a", x1.as_slice()),
        ("baseline_b", x2.as_slice()),
    ])
    .unwrap();
    let direct = antecedent::estimate::ancova::fit_ancova(&y, &assigned, &[&x1, &x2]).unwrap();
    let interval = antecedent_estimate::ancova::calibrated_bernoulli_interval_95(&direct, 0.5);
    let fit = run(
        &data,
        bernoulli(n, assigned.clone(), vec![0.5; n])
            .with_ancova(vec![VariableId::from_raw(1), VariableId::from_raw(2)]),
    );
    assert_eq!(fit.effect.to_bits(), direct.effect.to_bits());
    assert_eq!(fit.variance_upper_bound.to_bits(), direct.hc0_variance.to_bits());
    assert_eq!(fit.standard_error.unwrap().to_bits(), direct.hc0_variance.sqrt().to_bits());
    assert!(interval.is_some());
    same_interval(fit.interval_95, interval);
    assert_eq!(fit.treatment_units, direct.treated);
    assert_eq!(fit.control_units, direct.control);
}

/// Stratified randomization: blocked difference in means with blockwise Neyman variances.
#[test]
fn stratified_matches_the_direct_kernel() {
    let n = 120;
    let assigned = (0..n).map(|i| (i % 30) % 2 == 0).collect::<Vec<_>>();
    let y = (0..n)
        .map(|i| {
            1.0 + 2.0 * f64::from(assigned[i]) + 0.5 * (i as f64 * 0.3).sin() + (i / 30) as f64
        })
        .collect::<Vec<_>>();
    let blocks = (0..n).map(|i| Arc::<str>::from(format!("block-{}", i / 30))).collect::<Vec<_>>();
    let data = TabularData::from_f64_columns([("outcome", y.as_slice())]).unwrap();
    let reference_assignment = assigned.clone();
    let direct =
        antecedent::estimate::randomized_neyman::stratified_neyman_effect(&y, &assigned, &blocks)
            .unwrap();
    let query = RandomizedEffectQuery::with_design(
        RandomizationDesign::Stratified {
            blocks: Arc::from(blocks.clone()),
            treated_per_row: Arc::from([15; 120]),
        },
        VariableId::from_raw(0),
        assigned,
        vec![0.5; n],
        labels("unit", n),
        labels("row", n),
        ("control", "treated"),
    );
    let fit = run(&data, query);
    assert_eq!(fit.effect.to_bits(), direct.effect.to_bits());
    assert_eq!(fit.variance_upper_bound.to_bits(), direct.variance.to_bits());
    assert_eq!(direct.strata, 4);
    assert_eq!(fit.blocks.len(), n);
    // The blocked interval of the same cells, where the design supports one.
    let cells_owned = (0..4)
        .map(|block| {
            let rows = block * 30..block * 30 + 30;
            let arm = |treated: bool| {
                rows.clone()
                    .filter(|&i| reference_assignment[i] == treated)
                    .map(|i| y[i])
                    .collect::<Vec<_>>()
            };
            (arm(true), arm(false))
        })
        .collect::<Vec<_>>();
    let cells = cells_owned.iter().map(|(t, c)| (t.as_slice(), c.as_slice())).collect::<Vec<_>>();
    let blocked = antecedent_estimate::randomized_neyman::blocked_unit_itt(&cells).unwrap();
    assert!(blocked.interval_95.is_some());
    assert_eq!(
        fit.standard_error.unwrap().to_bits(),
        blocked.variance_upper_bound.sqrt().to_bits()
    );
    same_interval(fit.interval_95, blocked.interval_95);
}

/// Switchback: the compiled route runs `switchback_itt` on the sequence labels.
#[test]
fn switchback_matches_the_direct_kernel() {
    let (sequences, periods, probability) = (40_usize, 20_usize, 0.5);
    let n = sequences * periods;
    let assigned =
        (0..n).map(|i| (i.wrapping_mul(137) + (i / periods) * 31) % 11 < 5).collect::<Vec<_>>();
    let y = (0..n)
        .map(|i| {
            let (sequence, period) = (i / periods, i % periods);
            1.0 + 0.5 * (sequence as f64 * 0.7).sin()
                + 0.3 * (period as f64 * 0.4).cos()
                + if assigned[i] { 1.4 } else { 0.0 }
        })
        .collect::<Vec<_>>();
    let sequence_ids =
        (0..n).map(|i| Arc::<str>::from(format!("s{}", i / periods))).collect::<Vec<_>>();
    let period_ids =
        (0..n).map(|i| Arc::<str>::from(format!("p{}", i % periods))).collect::<Vec<_>>();
    let probabilities = vec![probability; n];
    let ids = sequence_ids.iter().map(AsRef::as_ref).collect::<Vec<&str>>();
    let direct =
        antecedent::estimate::switchback::switchback_itt(&y, &assigned, &probabilities, &ids)
            .unwrap();
    let data = TabularData::from_f64_columns([("outcome", y.as_slice())]).unwrap();
    let query = RandomizedEffectQuery::with_design(
        RandomizationDesign::Switchback { periods: period_ids.into() },
        VariableId::from_raw(0),
        assigned,
        probabilities,
        sequence_ids,
        labels("row", n),
        ("off", "on"),
    );
    let fit = run(&data, query);
    assert_eq!(fit.effect.to_bits(), direct.effect.to_bits());
    assert_eq!(fit.variance_upper_bound.to_bits(), direct.variance.to_bits());
    assert!(direct.interval_95.is_some());
    assert_eq!(fit.standard_error.unwrap().to_bits(), direct.variance.sqrt().to_bits());
    same_interval(fit.interval_95, direct.interval_95);
}

/// The probability-weighted exact test of a small Bernoulli design: the compiled route enumerates
/// the same allocations with the same kernel.
#[test]
fn bernoulli_exact_test_matches_the_direct_kernel() {
    let n = 12;
    let assigned = (0..n).map(|i| i % 3 != 0).collect::<Vec<_>>();
    let probabilities = (0..n).map(|i| 0.3 + 0.04 * i as f64).collect::<Vec<_>>();
    let y =
        (0..n).map(|i| 1.0 + 0.7 * (i as f64).sin() + f64::from(assigned[i])).collect::<Vec<_>>();
    let direct = antecedent::estimate::randomized_scores::bernoulli_exact_randomization_test(
        &y,
        &assigned,
        &probabilities,
    )
    .unwrap();
    let data = TabularData::from_f64_columns([("outcome", y.as_slice())]).unwrap();
    let fit = run(&data, bernoulli(n, assigned, probabilities).with_exact_randomization_test());
    assert_eq!(fit.effect.to_bits(), direct.observed.to_bits());
    assert_eq!(fit.randomization_p_value.unwrap().to_bits(), direct.p_value.to_bits());
    assert_eq!(fit.randomization_allocations, Some(direct.allocations as u64));
    assert_eq!(direct.allocations, 1 << n);
}

/// The Bernoulli exact test stays bounded to what can be enumerated and stays unadjusted.
#[test]
fn bernoulli_exact_test_refuses_what_the_direct_kernel_refuses() {
    let wide = bernoulli(21, (0..21).map(|i| i % 2 == 0).collect(), vec![0.5; 21]);
    assert!(wide.with_exact_randomization_test().validate().is_err());
    let narrow = bernoulli(20, (0..20).map(|i| i % 2 == 0).collect(), vec![0.5; 20]);
    assert!(narrow.clone().with_exact_randomization_test().validate().is_ok());
    assert!(
        narrow
            .with_exact_randomization_test()
            .with_fixed_cuped(VariableId::from_raw(1), 1.0)
            .validate()
            .is_err()
    );
}

/// Independent-Bernoulli 2x2 factorial: the compiled route is the four-cell multi-arm route over
/// the product cell probabilities. The two sum the same terms in a different order, so they agree
/// to rounding, not to the bit.
#[test]
fn factorial_agrees_with_the_four_cell_compiled_route() {
    let n = 400;
    let a = (0..n).map(|i| (i / 2) % 2 == 1).collect::<Vec<_>>();
    let b = (0..n).map(|i| i % 2 == 1).collect::<Vec<_>>();
    let (pa, pb) = (vec![0.5; n], vec![0.5; n]);
    let y = (0..n)
        .map(|i| {
            1.0 + 0.8 * f64::from(a[i])
                + 0.5 * f64::from(b[i])
                + 0.3 * f64::from(a[i] && b[i])
                + 0.4 * (0.21 * i as f64).sin()
        })
        .collect::<Vec<_>>();
    let direct =
        antecedent::estimate::factorial_bernoulli::estimate_factorial_2x2(&y, &a, &b, &pa, &pb)
            .unwrap();
    let cells = (0..n).map(|i| 2 * usize::from(a[i]) + usize::from(b[i])).collect::<Vec<_>>();
    let rows = (0..n)
        .map(|i| {
            [false, true]
                .into_iter()
                .flat_map(|u| [false, true].into_iter().map(move |v| (u, v)))
                .map(|(u, v)| {
                    (if u { pa[i] } else { 1.0 - pa[i] }) * (if v { pb[i] } else { 1.0 - pb[i] })
                })
                .collect::<Vec<f64>>()
        })
        .collect::<Vec<_>>();
    let assigned_probability = (0..n).map(|i| rows[i][cells[i]]).collect::<Vec<_>>();
    let query = RandomizedEffectQuery::with_design(
        RandomizationDesign::MultiArm {
            assignment: Arc::from(cells.clone()),
            probabilities: rows.into(),
            arms: ["00", "01", "10", "11"].map(Arc::<str>::from).into(),
        },
        VariableId::from_raw(0),
        cells.iter().map(|cell| *cell != 0).collect::<Vec<_>>(),
        assigned_probability,
        labels("unit", n),
        labels("row", n),
        ("00", "01"),
    );
    let data = TabularData::from_f64_columns([("outcome", y.as_slice())]).unwrap();
    let fit = run(&data, query);
    let close = |left: f64, right: f64| (left - right).abs() <= 1e-12 * left.abs().max(1.0);
    assert_eq!(fit.multi_arm_values.len(), 4);
    for (index, (_, value, variance, support)) in fit.multi_arm_values.iter().enumerate() {
        assert!(close(*value, direct.cell_means[index]), "cell {index} mean");
        assert_eq!(*support, direct.cell_support[index]);
        assert!(variance.is_finite());
    }
    let means = fit.multi_arm_values.iter().map(|(_, value, _, _)| *value).collect::<Vec<_>>();
    assert!(close(0.5 * (means[2] + means[3] - means[0] - means[1]), direct.factor_a_effect));
    assert!(close(means[0] - means[1] - means[2] + means[3], direct.interaction_effect));
}
