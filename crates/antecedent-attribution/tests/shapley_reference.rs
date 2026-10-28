//! Exact and permutation-sampled Shapley estimation against rational-arithmetic references.
//!
//! The frozen games in `oracle_data/shapley_reference.json` come from
//! `oracle_data/shapley_reference.py`, which computes each Shapley value in exact rational
//! arithmetic by two independent definitions (all player orders and the weighted subset sum).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_attribution::{
    AttributionError, Coalition, CoalitionPayoff, ShapleyEstimate, estimate_shapley,
};
use antecedent_core::{ComponentId, ExecutionContext, Parallelism, ShapleyConfig};
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    games: Vec<Game>,
}

#[derive(Deserialize)]
struct Game {
    name: String,
    n: usize,
    values: Vec<f64>,
    shapley: Vec<f64>,
    marginal_variance: Option<Vec<f64>>,
}

struct TablePayoff<'a>(&'a [f64]);

impl CoalitionPayoff for TablePayoff<'_> {
    fn value(&mut self, coalition: &Coalition) -> Result<f64, AttributionError> {
        Ok(self.0[coalition.to_bits().expect("test games have at most 12 players") as usize])
    }
}

fn fixture() -> Fixture {
    serde_json::from_str(include_str!("oracle_data/shapley_reference.json")).unwrap()
}

fn players(n: usize) -> Vec<ComponentId> {
    (0..n as u32).map(ComponentId::from_raw).collect()
}

fn run(game: &Game, config: &ShapleyConfig, threads: bool) -> ShapleyEstimate {
    let mut ctx = ExecutionContext::for_tests(3);
    if !threads {
        ctx.parallelism = Parallelism::serial();
    }
    estimate_shapley(&players(game.n), config, &mut TablePayoff(&game.values), &ctx).unwrap()
}

/// Largest absolute magnitude among the table entries, the natural scale of rounding error.
fn scale(game: &Game) -> f64 {
    game.values.iter().fold(0.0_f64, |m, v| m.max(v.abs()))
}

#[test]
fn exact_mode_matches_rational_shapley_on_every_game() {
    for game in &fixture().games {
        for threads in [false, true] {
            let est = run(game, &ShapleyConfig::exact(), threads);
            // Weights s!(n-s-1)!/n! are rounded once and 2^(n-1) marginals are summed with
            // compensation, so the error is a few ulps of the largest table entry, times n.
            let tol = 64.0 * f64::EPSILON * scale(game) * game.n as f64;
            for (i, (got, want)) in est.values.iter().zip(&game.shapley).enumerate() {
                assert!(
                    (got - want).abs() <= tol,
                    "{} player {i} threads={threads}: got {got} want {want} tol {tol}",
                    game.name
                );
            }
            // Efficiency: the sum is v(N) - v(empty) up to the same rounding.
            let residual = est.efficiency_residual.unwrap();
            assert!(residual.abs() <= tol, "{} efficiency residual {residual}", game.name);
            assert_eq!(est.budget.evaluations, 1u64 << game.n, "{}", game.name);
        }
    }
}

#[test]
fn sampled_modes_are_efficient_per_permutation_and_unbiased() {
    for game in fixture().games.iter().filter(|g| g.marginal_variance.is_some() && g.n >= 2) {
        let samples = 20_000usize;
        let est = run(game, &ShapleyConfig::permutation(samples).with_seed(17), false);
        let v_full = game.values[(1usize << game.n) - 1];
        // Every permutation's marginals telescope to v(N) - v(empty), so the mean does too.
        let total: f64 = est.values.iter().sum();
        let tol = 1e-9 * scale(game).max(1.0);
        assert!((total - (v_full - game.values[0])).abs() <= tol, "{} efficiency", game.name);
        let stderrs = est.component_mc_stderr.clone().unwrap();
        let variance = game.marginal_variance.as_ref().unwrap();
        for i in 0..game.n {
            // The sample mean lies within five reported standard errors of the exact value.
            let se = stderrs[i];
            assert!(
                (est.values[i] - game.shapley[i]).abs() <= 5.0 * se + 1e-12,
                "{} player {i}: {} vs {} (se {se})",
                game.name,
                est.values[i],
                game.shapley[i]
            );
            // The reported standard error is the exact permutation sd over sqrt(samples).
            let want_se = (variance[i] / samples as f64).sqrt();
            assert!(
                (se - want_se).abs() <= 0.05 * want_se + 1e-12,
                "{} player {i}: stderr {se} vs exact {want_se}",
                game.name
            );
        }
    }
}

#[test]
fn null_player_and_symmetric_players_are_recognized() {
    let fixture = fixture();
    let game = fixture.games.iter().find(|g| g.name == "glove_null_player").unwrap();
    let est = run(game, &ShapleyConfig::exact(), false);
    // Players 0 and 1 are interchangeable left gloves; player 3 never changes the value.
    assert!((est.values[0] - est.values[1]).abs() <= 1e-15);
    assert!(est.values[3].abs() <= 1e-15);
    assert!((est.values[2] - 2.0 / 3.0).abs() <= 1e-15);
}

#[test]
fn single_player_receives_the_whole_difference() {
    let fixture = fixture();
    let game = fixture.games.iter().find(|g| g.name == "random_1").unwrap();
    let est = run(game, &ShapleyConfig::exact(), false);
    assert_eq!(est.values[0], game.values[1] - game.values[0]);
}
