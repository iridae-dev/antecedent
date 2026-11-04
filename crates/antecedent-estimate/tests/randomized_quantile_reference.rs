//! The randomized quantile treatment effect against a design small enough to solve by hand.
//!
//! The design has forty rows. The control arm carries the outcomes 1, 2, ..., 20 and the treated
//! arm 3, 4, ..., 22, so each arm is uniform on twenty integers and the treated law is the
//! control law shifted by two. Everything below is derived on paper from that table and is
//! compared with the library; nothing is read back from it.
//!
//! Declared grid 0, 5, 10, 15, 20, 25 at tau = 1/2. The control distribution function at the
//! grid is 0, 1/4, 1/2, 3/4, 1, 1 and the treated one 0, 3/20, 2/5, 13/20, 9/10, 1. The control
//! level 1/2 is reached exactly at the grid point 10 (bracket weight 1), so Q0 = 10; the treated
//! level lies a fraction (1/2 - 2/5) / (13/20 - 2/5) = 2/5 of the way across the bracket
//! [10, 15], so Q1 = 12. Both brackets have density 1/20 (rise 1/4 over width 5), which times the
//! span 25 is 5/4, above the guard. The effect is 2 to the tolerance of the floating bracket
//! weight, which is 1e-12.
//!
//! The row influence of the ratio estimator of an arm distribution function at a threshold is
//! `(1{T = a} / p_a)(1{Y <= c} - F_a(c))` with `p_a = 1/2`. The control quantile influence is
//! minus that over the density: -20 for the ten control rows at or below 10 and +20 for the other
//! ten. The treated influence blends the thresholds 10 and 15 with weights 3/5 and 2/5 before
//! dividing by -1/20: -20 for the eight rows at or below 10, +4 for the five between 10 and 15 and
//! +20 for the seven above 15. The difference influence is the treated influence on treated rows
//! and minus the control influence on control rows, so its sum of squares is
//! 8 * 400 + 5 * 16 + 7 * 400 + 20 * 400 = 14 080, it has mean zero, and the iid standard error is
//! `sqrt(14080 / (39 * 40))`. The interval is the estimate plus and minus 1.959963984540054
//! times that standard error, the 97.5% standard normal point written out here as a literal.
//!
//! Data-driven grid. The pooled sorted outcomes put the 19 empirical grid points on the integers
//! 3, 4, ..., 20 (the tenth and eleventh points coincide and are merged). On that grid
//! F0(c) = c / 20 and F1(c) = (c - 2) / 20, so Q0 = 10, Q1 = 12 and every row influence is +-20, giving
//! the standard error `20 / sqrt(39)`. At tau = 1/4 the quantiles are 5 and 7 and at tau = 3/4
//! they are 15 and 17; there each arm's influence is -30 for the five rows at or below its
//! quantile and +10 for the fifteen above (tau = 1/4) or -10 for fifteen rows and +30 for five
//! (tau = 3/4), so the difference influence has sum of squares 2 * (5 * 900 + 15 * 100) = 12 000
//! and the standard error `sqrt(12000 / (39 * 40))`.
//!
//! The tolerance for the paper values is 1e-9 absolute: the only inexactness is the bracket
//! weight `(tau - F_lo) / (F_hi - F_lo)` in double precision, of relative size 1e-15.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_estimate::randomized_quantile::{
    randomized_quantile_effect, randomized_quantile_effect_on_grid,
};

const NORMAL_975: f64 = 1.959_963_984_540_054;

fn design() -> (Vec<f64>, Vec<bool>) {
    let mut outcome: Vec<f64> = (1..=20).map(f64::from).collect();
    outcome.extend((3..=22).map(f64::from));
    let treated: Vec<bool> = (0..40).map(|row| row >= 20).collect();
    (outcome, treated)
}

fn close(got: f64, want: f64, what: &str) {
    assert!((got - want).abs() < 1e-9, "{what}: {got} vs {want}");
}

#[test]
fn declared_grid_matches_the_hand_derivation_of_value_influence_and_interval() {
    let (outcome, treated) = design();
    let grid = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0];
    let effect = randomized_quantile_effect_on_grid(&outcome, &treated, &grid, 0.5).unwrap();
    close(effect.control_quantile, 10.0, "control quantile");
    close(effect.treated_quantile, 12.0, "treated quantile");
    close(effect.difference, 2.0, "difference");
    close(effect.densities[0], 0.05, "control density");
    close(effect.densities[1], 0.05, "treated density");
    assert_eq!((effect.n_control, effect.n_treated), (20, 20));

    // The paper influence, row by row.
    for (row, value) in effect.influence.iter().enumerate() {
        let y = outcome[row];
        let want = if row < 20 {
            // Minus the control quantile influence.
            if y <= 10.0 { 20.0 } else { -20.0 }
        } else if y <= 10.0 {
            -20.0
        } else if y <= 15.0 {
            4.0
        } else {
            20.0
        };
        close(*value, want, &format!("influence of row {row}"));
    }
    assert!(effect.influence.iter().sum::<f64>().abs() < 1e-9, "the influence is centered");

    // Interval construction: se = sqrt(sum of squares / ((n - 1) n)), interval = est -/+ z * se.
    let se = (14_080.0_f64 / (39.0 * 40.0)).sqrt();
    close(effect.se, se, "standard error");
    close(effect.interval_95[0], 2.0 - NORMAL_975 * se, "interval lower end");
    close(effect.interval_95[1], 2.0 + NORMAL_975 * se, "interval upper end");
}

#[test]
fn empirical_grid_recovers_the_location_shift_at_three_levels() {
    let (outcome, treated) = design();
    for (tau, control, active, sum_squares) in
        [(0.25, 5.0, 7.0, 12_000.0), (0.5, 10.0, 12.0, 16_000.0), (0.75, 15.0, 17.0, 12_000.0)]
    {
        let effect = randomized_quantile_effect(&outcome, &treated, tau).unwrap();
        assert_eq!(effect.thresholds.len(), 18, "the integers 3..=20");
        close(effect.control_quantile, control, "control quantile");
        close(effect.treated_quantile, active, "treated quantile");
        close(effect.difference, 2.0, "difference");
        close(effect.influence.iter().map(|v| v * v).sum::<f64>(), sum_squares, "sum of squares");
        let se = (sum_squares / (39.0 * 40.0_f64)).sqrt();
        close(effect.se, se, "standard error");
        close(effect.interval_95[0], 2.0 - NORMAL_975 * se, "interval lower end");
        close(effect.interval_95[1], 2.0 + NORMAL_975 * se, "interval upper end");
    }
}

#[test]
fn unsupported_inputs_are_refused() {
    let (outcome, treated) = design();
    let grid = [0.0, 5.0, 10.0, 15.0];
    // The control distribution function stops at 3/4 on this grid, below the level 0.9.
    assert!(randomized_quantile_effect_on_grid(&outcome, &treated, &grid, 0.9).is_err());
    // The level must lie in (0, 1).
    assert!(randomized_quantile_effect(&outcome, &treated, 1.0).is_err());
    assert!(randomized_quantile_effect(&outcome, &treated, 0.0).is_err());
    // An arm of five rows carries no estimable distribution function.
    let mut small = vec![false; 40];
    for flag in small.iter_mut().take(5) {
        *flag = true;
    }
    assert!(randomized_quantile_effect(&outcome, &small, 0.5).is_err());
    // Misaligned and non-finite rows.
    assert!(randomized_quantile_effect(&outcome[..39], &treated, 0.5).is_err());
    let mut bad = outcome.clone();
    bad[3] = f64::NAN;
    assert!(randomized_quantile_effect(&bad, &treated, 0.5).is_err());
}
