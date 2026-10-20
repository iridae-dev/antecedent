//! Switchback posterior against an independent whitened-least-squares derivation.
//!
//! Reference: the full block-diagonal error covariance is Cholesky-whitened and the
//! Normal-Inverse-Gamma prior is written as pseudo-observations, so the posterior mean is the
//! least-squares solution of the stacked system and `b_n = b0 + RSS / 2`; quantiles are those of
//! `scipy.stats.t`. Four sequences of four periods; effect precision 0.5, nuisance precision
//! 0.01, `a0 = 2`, `b0 = 1`.

use antecedent_estimate::gaussian_linear_posterior::PosteriorSummary;
use antecedent_estimate::switchback::switchback_itt;
use antecedent_estimate::switchback_posterior::{
    CarryoverDeclaration, PeriodAdjustment, SequenceDependence, SwitchbackModel, SwitchbackPrior,
    SwitchbackRows, posterior_switchback,
};

#[path = "common/checks.rs"]
mod checks;

fn close(actual: f64, expected: f64, what: &str) {
    assert!((actual - expected).abs() <= 1e-7, "{what}: {actual} vs {expected}");
}

fn check(summary: &PosteriorSummary, expected: [f64; 4], what: &str) {
    // Location, 5% and 95% quantiles, P(> 0); the degrees of freedom are 2 * (2 + 16 / 2).
    close(summary.mean, expected[0], &format!("{what} mean"));
    close(summary.degrees_of_freedom, 20.0, &format!("{what} df"));
    close(summary.interval[0], expected[1], &format!("{what} 5%"));
    close(summary.interval[1], expected[2], &format!("{what} 95%"));
    close(summary.probability_positive, expected[3], &format!("{what} P(> 0)"));
}

const ASSIGNMENT: [[bool; 4]; 4] = [
    [true, false, true, false],
    [false, true, false, true],
    [true, true, false, false],
    [false, false, true, true],
];
const OUTCOME: [[f64; 4]; 4] =
    [[2.3, 1.1, 2.9, 0.8], [0.7, 2.2, 1.4, 2.6], [2.8, 3.1, 0.9, 1.3], [1.2, 0.6, 2.4, 3.0]];

struct Fixture {
    outcomes: Vec<f64>,
    assignment: Vec<bool>,
    ids: Vec<String>,
}

fn fixture() -> Fixture {
    let mut f = Fixture { outcomes: vec![], assignment: vec![], ids: vec![] };
    for s in 0..4 {
        for t in 0..4 {
            f.outcomes.push(OUTCOME[s][t]);
            f.assignment.push(ASSIGNMENT[s][t]);
            f.ids.push(format!("s{s}"));
        }
    }
    f
}

fn model(
    carryover: CarryoverDeclaration,
    dependence: SequenceDependence,
    periods: PeriodAdjustment,
) -> SwitchbackModel {
    SwitchbackModel {
        carryover,
        dependence,
        periods,
        prior: SwitchbackPrior {
            effect_precision: 0.5,
            nuisance_precision: 0.01,
            variance_shape: 2.0,
            variance_scale: 1.0,
        },
    }
}

fn run(
    model: &SwitchbackModel,
) -> Result<
    antecedent_estimate::switchback_posterior::SwitchbackPosterior,
    antecedent_estimate::EstimationError,
> {
    let f = fixture();
    let ids: Vec<&str> = f.ids.iter().map(String::as_str).collect();
    posterior_switchback(
        &SwitchbackRows { outcomes: &f.outcomes, assignment: &f.assignment, sequence_ids: &ids },
        model,
        0.9,
    )
}

#[test]
fn compound_symmetric_without_carryover_matches_whitened_least_squares() {
    let posterior = run(&model(
        CarryoverDeclaration::None,
        SequenceDependence::CompoundSymmetric { rho: 0.4 },
        PeriodAdjustment::None,
    ))
    .unwrap();
    let (shape, scale) = posterior.variance_shape_and_scale;
    close(shape, 10.0, "a_n");
    close(scale, 2.685_571_931_631, "b_n");
    check(
        &posterior.direct_effect,
        [1.547_248_493, 1.213_436_207, 1.881_060_779, 0.999_999_941],
        "direct",
    );
    assert!(posterior.carryover_effect.is_none() && posterior.period_contrasts.is_empty());
    assert_eq!(posterior.evidence.sequences, 4);
    assert_eq!(posterior.evidence.rows, 16);
}

#[test]
fn autoregressive_first_order_carryover_reports_direct_carryover_and_sustained_effects() {
    let posterior = run(&model(
        CarryoverDeclaration::FirstOrder,
        SequenceDependence::Autoregressive { rho: 0.5 },
        PeriodAdjustment::None,
    ))
    .unwrap();
    close(posterior.variance_shape_and_scale.1, 2.737_124_349_379, "b_n");
    check(
        &posterior.direct_effect,
        [1.570_684_728, 1.204_478_494, 1.936_890_962, 0.999_999_809],
        "direct",
    );
    check(
        &posterior.carryover_effect.unwrap(),
        [0.154_196_445, -0.217_349_523, 0.525_742_412, 0.758_799_573],
        "carryover",
    );
    // The sustained effect uses the posterior covariance of the two coefficients.
    check(
        &posterior.sustained_effect.unwrap(),
        [1.724_881_173, 1.091_407_800, 2.358_354_546, 0.999_930_685],
        "sustained",
    );
}

#[test]
fn period_effects_report_contrasts_against_the_first_period() {
    let posterior = run(&model(
        CarryoverDeclaration::None,
        SequenceDependence::Independent,
        PeriodAdjustment::FixedEffects,
    ))
    .unwrap();
    check(
        &posterior.direct_effect,
        [1.478_898_724, 1.092_020_496, 1.865_776_953, 0.999_998_992],
        "direct",
    );
    let expected = [
        [0.001_694_708, -0.576_266_349, 0.579_655_766, 0.501_992_490],
        [0.151_320_643, -0.426_640_414, 0.729_281_701, 0.671_778_576],
        [0.176_258_299, -0.401_702_758, 0.754_219_357, 0.697_656_902],
    ];
    assert_eq!(posterior.period_contrasts.len(), 3);
    for (j, (summary, e)) in posterior.period_contrasts.iter().zip(expected).enumerate() {
        check(summary, e, &format!("period {}", j + 1));
    }
}

#[test]
fn carryover_with_period_effects_and_dependence_together() {
    let posterior = run(&model(
        CarryoverDeclaration::FirstOrder,
        SequenceDependence::Autoregressive { rho: 0.5 },
        PeriodAdjustment::FixedEffects,
    ))
    .unwrap();
    check(
        &posterior.direct_effect,
        [1.576_260_652, 1.204_439_637, 1.948_081_667, 0.999_999_773],
        "direct",
    );
    check(
        &posterior.sustained_effect.unwrap(),
        [1.742_417_907, 1.063_801_891, 2.421_033_924, 0.999_870_744],
        "sustained",
    );
    close(posterior.period_contrasts[0].mean, -0.081_948_464, "period 1");
    close(posterior.period_contrasts[2].interval[1], 0.715_850_050, "period 3 95%");
}

#[test]
fn balanced_noisy_schedule_agrees_with_the_horvitz_thompson_effect_when_the_effect_is_constant() {
    // 32 sequences of 12 periods, alternating assignment; y = 1.0 * A + deterministic noise.
    let n = 32 * 12;
    let assignment: Vec<bool> = (0..n).map(|i| i % 2 == 0).collect();
    let outcomes: Vec<f64> = (0..n)
        .map(|i| {
            let noise = ((i * 7919 + 13) % 17) as f64 / 17.0 - 0.5;
            f64::from(u8::from(assignment[i])) + 0.4 * noise
        })
        .collect();
    let ids: Vec<String> = (0..n).map(|i| format!("s{:02}", i / 12)).collect();
    let refs: Vec<&str> = ids.iter().map(String::as_str).collect();
    let ht = switchback_itt(&outcomes, &assignment, &vec![0.5; n], &refs).unwrap();
    let weak = SwitchbackModel {
        carryover: CarryoverDeclaration::None,
        dependence: SequenceDependence::Independent,
        periods: PeriodAdjustment::None,
        prior: SwitchbackPrior {
            effect_precision: 1e-6,
            nuisance_precision: 1e-6,
            variance_shape: 1.0,
            variance_scale: 1.0,
        },
    };
    let posterior = posterior_switchback(
        &SwitchbackRows { outcomes: &outcomes, assignment: &assignment, sequence_ids: &refs },
        &weak,
        0.95,
    )
    .unwrap();
    assert!((posterior.direct_effect.mean - ht.effect).abs() < 1e-3);
    assert!((posterior.direct_effect.mean - 1.0).abs() < 0.05);
    let [lo, hi] = posterior.direct_effect.interval;
    assert!(lo < 1.0 && 1.0 < hi, "{lo} {hi}");
}

#[test]
fn sequence_order_does_not_change_the_posterior() {
    let f = fixture();
    let model = model(
        CarryoverDeclaration::FirstOrder,
        SequenceDependence::CompoundSymmetric { rho: 0.2 },
        PeriodAdjustment::None,
    );
    let ids: Vec<&str> = f.ids.iter().map(String::as_str).collect();
    let forward = posterior_switchback(
        &SwitchbackRows { outcomes: &f.outcomes, assignment: &f.assignment, sequence_ids: &ids },
        &model,
        0.9,
    )
    .unwrap();
    // Reverse the order of the four sequences, keeping each sequence in period order.
    let reorder = |v: &[f64]| -> Vec<f64> {
        (0..4).rev().flat_map(|s| v[4 * s..4 * s + 4].to_vec()).collect()
    };
    let outcomes = reorder(&f.outcomes);
    let assignment: Vec<bool> =
        reorder(&f.assignment.iter().map(|a| f64::from(u8::from(*a))).collect::<Vec<_>>())
            .iter()
            .map(|v| *v > 0.5)
            .collect();
    let reversed_ids: Vec<&str> =
        (0..4).rev().flat_map(|s| ids[4 * s..4 * s + 4].to_vec()).collect();
    let backward = posterior_switchback(
        &SwitchbackRows {
            outcomes: &outcomes,
            assignment: &assignment,
            sequence_ids: &reversed_ids,
        },
        &model,
        0.9,
    )
    .unwrap();
    close(forward.direct_effect.mean, backward.direct_effect.mean, "mean");
    close(forward.direct_effect.interval[1], backward.direct_effect.interval[1], "upper");
}

#[test]
fn undeclared_carryover_and_dependence_and_bad_designs_refuse_with_the_next_attempt() {
    let none = |dep| model(CarryoverDeclaration::None, dep, PeriodAdjustment::None);
    let (code, message) = checks::coded_refusal(
        run(&model(
            CarryoverDeclaration::Unrestricted,
            SequenceDependence::Independent,
            PeriodAdjustment::None,
        ))
        .unwrap_err(),
    );
    assert_eq!(code, "cell_not_licensed");
    assert!(message.contains("switchback_posterior.carryover_unrestricted"), "{message}");
    assert!(message.contains("Next attempt") && message.contains("washout"), "{message}");

    let (code, message) =
        checks::coded_refusal(run(&none(SequenceDependence::Unspecified)).unwrap_err());
    assert_eq!(code, "cell_not_licensed");
    assert!(message.contains("switchback_posterior.dependence_undeclared"), "{message}");
    assert!(message.contains("Next attempt"), "{message}");

    // Compound symmetry below -1 / (m - 1) = -1/3 is not a covariance matrix.
    let (_, message) = checks::coded_refusal(
        run(&none(SequenceDependence::CompoundSymmetric { rho: -0.5 })).unwrap_err(),
    );
    assert!(message.contains("switchback_posterior.invalid_correlation"), "{message}");
    let (_, message) = checks::coded_refusal(
        run(&none(SequenceDependence::Autoregressive { rho: 1.0 })).unwrap_err(),
    );
    assert!(message.contains("switchback_posterior.invalid_correlation"), "{message}");

    let f = fixture();
    let ids: Vec<&str> = f.ids.iter().map(String::as_str).collect();
    let all_treated = vec![true; 16];
    let (_, message) = checks::coded_refusal(
        posterior_switchback(
            &SwitchbackRows { outcomes: &f.outcomes, assignment: &all_treated, sequence_ids: &ids },
            &none(SequenceDependence::Independent),
            0.9,
        )
        .unwrap_err(),
    );
    assert!(message.contains("switchback_posterior.single_arm"), "{message}");

    // Unequal sequence lengths cannot carry period effects.
    let mut uneven = ids.clone();
    uneven[0] = "s1";
    let (_, message) = checks::coded_refusal(
        posterior_switchback(
            &SwitchbackRows {
                outcomes: &f.outcomes,
                assignment: &f.assignment,
                sequence_ids: &uneven,
            },
            &model(
                CarryoverDeclaration::None,
                SequenceDependence::Independent,
                PeriodAdjustment::FixedEffects,
            ),
            0.9,
        )
        .unwrap_err(),
    );
    assert!(message.contains("switchback_posterior.unbalanced_periods"), "{message}");

    let one_sequence = vec!["only"; 16];
    let (_, message) = checks::coded_refusal(
        posterior_switchback(
            &SwitchbackRows {
                outcomes: &f.outcomes,
                assignment: &f.assignment,
                sequence_ids: &one_sequence,
            },
            &none(SequenceDependence::Independent),
            0.9,
        )
        .unwrap_err(),
    );
    assert!(message.contains("switchback_posterior.too_few_sequences"), "{message}");
}
