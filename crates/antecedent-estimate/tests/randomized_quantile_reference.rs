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
//! [10, 15], so Q1 = 12. Both grid brackets have slope 1/20 (rise 1/4 over width 5), which times
//! the span 25 is 5/4, above the inversion guard. The effect is 2 to the tolerance of the
//! floating bracket weight, which is 1e-12.
//!
//! The density dividing the influence is not that bracket slope but each arm's Siddiqui sparsity
//! estimate `2h / (Q_a(tau + h) - Q_a(tau - h))` with the type-7 empirical quantile of the arm
//! and the Hall-Sheather bandwidth `h = (z^2 * 1.5 phi(x)^2 / (2 x^2 + 1) / n_a)^(1/3)`,
//! `x = Phi^-1(tau)`, `z = 1.959963984540054`, capped at `0.9 * min(tau, 1 - tau)`. At tau = 1/2,
//! `x = 0` and `phi(0)^2 = 1 / (2 pi)`, so with `n_a = 20`
//! `h = (3.8414588206941254 * 0.238732414637843 / 20)^(1/3) = 0.0458540...^(1/3) = 0.357925...`,
//! below the cap 0.45. Each arm is twenty equally spaced integers, so its type-7 quantile is
//! linear on [0, 1] (`Q0(p) = 1 + 19 p`, zero-based rank `19 p`, and `Q1(p) = 3 + 19 p`); the
//! spread is `19 * 2h` and the density is `2h / (38 h) = 1/19` in both arms, whatever `h` is
//! (as long as `tau -/+ h` stays in [0, 1]). The density does not depend on the grid.
//!
//! The row influence of the ratio estimator of an arm distribution function at a threshold is
//! `(1{T = a} / p_a)(1{Y <= c} - F_a(c))` with `p_a = 1/2`, blended across the inversion bracket
//! with the bracket weight (the grid-interpolated numerator), and the quantile influence is minus
//! that over the density 1/19. Control: the blend is the threshold 10 alone, `+-1`, so -19 for
//! the ten control rows at or below 10 and +19 for the other ten. Treated: weights 3/5 and 2/5 on
//! the thresholds 10 and 15 give `(3/5)(1 - 2/5)/(1/2) + (2/5)(1 - 13/20)/(1/2) = 1` for the
//! eight rows at or below 10, `(3/5)(-2/5)/(1/2) + (2/5)(7/20)/(1/2) = -1/5` for the five
//! between 10 and 15 and `-1` for the seven above 15, so the influences are -19, +19/5 = 3.8 and
//! +19. The difference influence is the treated influence on treated rows and minus the control
//! influence on control rows, so its sum of squares is
//! 8 * 361 + 5 * 14.44 + 7 * 361 + 20 * 361 = 12 635 + 72.2 = 12 707.2, it has mean zero, and the
//! iid standard error is `sqrt(12707.2 / (39 * 40))`. The interval is the estimate plus and minus
//! 1.959963984540054 times that standard error, the 97.5% standard normal point written out here
//! as a literal.
//!
//! Data-driven grid. The pooled sorted outcomes put the 19 empirical grid points on the integers
//! 3, 4, ..., 20 (the tenth and eleventh points coincide and are merged). On that grid
//! F0(c) = c / 20 and F1(c) = (c - 2) / 20, so Q0 = 10, Q1 = 12, and the blended numerator is
//! `+-1` on every row; over the density 1/19 every row influence is +-19, so the sum of squares
//! is 40 * 361 = 14 440 and the standard error `19 / sqrt(39)`. At tau = 1/4 the quantiles are 5
//! and 7 and at tau = 3/4 they are 15 and 17. There the Hall-Sheather bandwidth is
//! `(3.8414588206941254 * 1.5 * phi(0.6744897...)^2 / (2 * 0.6744897...^2 + 1) / 20)^(1/3)
//! = 0.24789...`, above the cap `0.9 * 1/4 = 0.225`, so `h = 0.225` and the window is
//! [0.025, 0.475] (or [0.525, 0.975]), still inside [0, 1] where the type-7 quantile is linear,
//! so the density is again 1/19. The numerator is `(1 - 1/4)/(1/2) = 3/2` for the five rows at
//! or below the arm's quantile and `-1/2` for the fifteen above (tau = 1/4), or `1/2` for fifteen
//! rows and `-3/2` for five (tau = 3/4); over 1/19 each arm's influence is -28.5 for five rows
//! and +9.5 for fifteen (tau = 1/4) or -9.5 for fifteen and +28.5 for five (tau = 3/4), so the
//! difference influence has sum of squares 2 * (5 * 812.25 + 15 * 90.25) = 10 830 and the
//! standard error `sqrt(10830 / (39 * 40))`.
//!
//! The tolerance for the paper values is 1e-9 absolute: the only inexactness is the bracket
//! weight `(tau - F_lo) / (F_hi - F_lo)` and the cancellation of `h` in `2h / (38 h)` in double
//! precision, each of relative size 1e-15.
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
    // Siddiqui sparsity densities, not the bracket slope 1/20.
    close(effect.densities[0], 1.0 / 19.0, "control density");
    close(effect.densities[1], 1.0 / 19.0, "treated density");
    assert_eq!((effect.n_control, effect.n_treated), (20, 20));

    // The paper influence, row by row.
    for (row, value) in effect.influence.iter().enumerate() {
        let y = outcome[row];
        let want = if row < 20 {
            // Minus the control quantile influence.
            if y <= 10.0 { 19.0 } else { -19.0 }
        } else if y <= 10.0 {
            -19.0
        } else if y <= 15.0 {
            3.8
        } else {
            19.0
        };
        close(*value, want, &format!("influence of row {row}"));
    }
    assert!(effect.influence.iter().sum::<f64>().abs() < 1e-9, "the influence is centered");

    // Interval construction: se = sqrt(sum of squares / ((n - 1) n)), interval = est -/+ z * se.
    let se = (12_707.2_f64 / (39.0 * 40.0)).sqrt();
    close(effect.se, se, "standard error");
    close(effect.interval_95[0], 2.0 - NORMAL_975 * se, "interval lower end");
    close(effect.interval_95[1], 2.0 + NORMAL_975 * se, "interval upper end");
}

#[test]
fn empirical_grid_recovers_the_location_shift_at_three_levels() {
    let (outcome, treated) = design();
    for (tau, control, active, sum_squares) in
        [(0.25, 5.0, 7.0, 10_830.0), (0.5, 10.0, 12.0, 14_440.0), (0.75, 15.0, 17.0, 10_830.0)]
    {
        let effect = randomized_quantile_effect(&outcome, &treated, tau).unwrap();
        assert_eq!(effect.thresholds.len(), 18, "the integers 3..=20");
        close(effect.control_quantile, control, "control quantile");
        close(effect.treated_quantile, active, "treated quantile");
        close(effect.difference, 2.0, "difference");
        close(effect.densities[0], 1.0 / 19.0, "control density");
        close(effect.densities[1], 1.0 / 19.0, "treated density");
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
