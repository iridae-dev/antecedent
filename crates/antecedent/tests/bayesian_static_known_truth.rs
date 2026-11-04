//! Known-truth evidence for the static Bayesian `AverageEffect` and `ConditionalEffect`
//! estimators on every coordinate they claim.
//!
//! * `bayesian.robust_ate`: the cross-fitted orthogonal ATE refits ridge (penalty 1)
//!   nuisances under Bayesian-bootstrap weights. On a design whose outcome is constant within
//!   each treatment arm (`y = 2 t`, `z` only causes `t`), every weighted ridge fit of a constant
//!   is that constant (the intercept is unpenalized), every residual vanishes, and every draw of
//!   the posterior equals the contrast 2. The posterior mean is therefore 2 to round-off.
//! * `bayesian.basis.gcomp`: the conjugate normal posterior of the quadratic-basis outcome
//!   model on the noise-free law `y = 2 t + z` (and `y = 2 t + z + t z` for the conditional
//!   effect). The prior scale 1000 shrinks the coefficients by less than 1e-6 relative to the
//!   Gram diagonal, the residual prior floors the posterior sd far below 1e-2, and the
//!   tolerance is six Monte-Carlo standard errors of a mean of at least 2 000 draws of a
//!   posterior of that sd plus the shrinkage bound.
//! * `bayesian.randomized_quantile`: the Dirichlet-multinomial posterior of the binned outcome
//!   law has closed-form CDF means at the grid, `(prior + cumulative count) / total mass`,
//!   which the test computes from the raw outcomes and compares at 1e-12 (no draws involved).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#[path = "known_truth_cells/mod.rs"]
mod support;

use antecedent::{BayesianConfig, DesignPosteriorModel, EstimatorId, EstimatorSpec, InferenceMode};
use antecedent_core::{
    AverageEffectQuery, CausalQuery, ConditionalEffectQuery, OutcomeFunctional, VariableId,
};
use antecedent_data::TabularData;
use std::sync::Arc;
use support::{
    Design, GraphKind, Structure, UNKNOWN_TRUTH, assert_coordinate, confounded_truth,
    for_each_matching, identified_atoms, on_large_stack, rows, run_average_effect, run_query,
    stratified_truth_with, table_with, unidentified_mass, unknown_table_with,
};

/// Mean of at least 2 000 posterior draws of a posterior with sd at most 1e-2, six standard
/// errors, plus the prior-shrinkage bound.
const DRAW_MEAN_TOLERANCE: f64 = 6.0 * 1e-2 / 44.721_359_549_995_79 + 1e-6;
/// Draws requested from the conjugate Bayesian cells.
const DRAWS: usize = 4_000;

fn conjugate(draws: usize) -> InferenceMode {
    InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(draws).prior_scale(1_000.0))
}

fn constant_arm(t: f64, _z: f64) -> f64 {
    2.0 * t
}

fn constant_arm_unknown(_era: f64, t: f64) -> f64 {
    t
}

// ------------------------------------------------------------------ bayesian.robust_ate

#[test]
fn bayesian_robust_ate_matches_the_back_door_contrast_on_every_claimed_coordinate() {
    let truth = stratified_truth_with(constant_arm);
    assert!((truth - 2.0).abs() < 1e-12, "design truth {truth}");
    on_large_stack(move || {
        let design = Design {
            main: table_with(constant_arm),
            unknown: unknown_table_with(constant_arm_unknown),
            ..Design::confounded()
        };
        let spec = EstimatorSpec::Default(EstimatorId::BayesianRobustAte);
        let inference =
            InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(100).prior_scale(1_000.0));
        for_each_matching(
            "bayesian.robust_ate|AverageEffect",
            |c| {
                !matches!(c.graph, GraphKind::CoDetermined | GraphKind::Unknown)
                    && c.suite_label == "none"
            },
            |coordinate| {
                let result = run_average_effect(coordinate, &spec, &inference, &design, 41);
                assert_coordinate(
                    coordinate,
                    "bayesian.robust_ate",
                    &result,
                    truth,
                    &UNKNOWN_TRUTH,
                    1e-6,
                );
            },
        );
    });
}

// ------------------------------------------------------------------ bayesian.basis.gcomp

#[test]
fn bayesian_basis_gcomp_average_effect_matches_the_back_door_contrast_on_every_coordinate() {
    let truth = confounded_truth();
    on_large_stack(move || {
        let design = Design::confounded();
        let spec = EstimatorSpec::Default(EstimatorId::BayesianBasisGcomp);
        let inference = conjugate(DRAWS);
        for_each_matching(
            "bayesian.basis.gcomp|AverageEffect",
            |c| c.graph == GraphKind::Dag && c.suite_label == "none",
            |coordinate| {
                let result = run_average_effect(coordinate, &spec, &inference, &design, 43);
                assert_coordinate(
                    coordinate,
                    "bayesian.basis.gcomp",
                    &result,
                    truth,
                    &UNKNOWN_TRUTH,
                    DRAW_MEAN_TOLERANCE,
                );
                // The Monte-Carlo bound assumes a posterior sd below 1e-2 (4 000 draws >= 2 000).
                let posterior = result.posterior.as_ref().expect("posterior");
                let column = posterior.effect_column().unwrap();
                assert!(posterior.summaries.sd[column] < 1e-2, "{}: posterior sd", coordinate.text);
            },
        );
    });
}

/// `y = 2 t + z + t z`: the effect of `t` at modifier value `z` is `2 + z`.
fn interaction_law(t: f64, z: f64) -> f64 {
    2.0 * t + z + t * z
}

#[test]
fn bayesian_basis_gcomp_conditional_effect_matches_the_modifier_specific_effect() {
    // tau(z) = 2 + z, and the modifier is balanced (60 rows at each of -1 and 1), so the
    // modifier-averaged conditional effect is 2.
    let averaged = rows().iter().map(|row| 2.0 + row.z).sum::<f64>() / rows().len() as f64;
    assert!((averaged - 2.0).abs() < 1e-12);
    on_large_stack(move || {
        let design = Design { main: table_with(interaction_law), ..Design::confounded() };
        let spec = EstimatorSpec::Default(EstimatorId::BayesianBasisGcomp);
        let inference = conjugate(DRAWS);
        let query = CausalQuery::ConditionalEffect(
            ConditionalEffectQuery::try_new(
                AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1))
                    .with_effect_modifiers([VariableId::from_raw(2)]),
            )
            .unwrap(),
        );
        for_each_matching(
            "bayesian.basis.gcomp|ConditionalEffect",
            |c| c.suite_label == "none",
            |coordinate| {
                let result =
                    run_query(coordinate, &spec, &inference, &design, query.clone(), None, 47);
                if coordinate.structure == Structure::GraphPosterior {
                    let atoms = identified_atoms(&result);
                    assert_eq!(atoms.len(), 2, "{}", coordinate.text);
                    for atom in &atoms {
                        assert!(
                            (atom[0] - averaged).abs() < DRAW_MEAN_TOLERANCE,
                            "{}: atom {} vs {averaged}",
                            coordinate.text,
                            atom[0]
                        );
                    }
                    assert!(unidentified_mass(&result) < 1e-12, "{}", coordinate.text);
                } else {
                    assert_eq!(
                        result.logical_plan.estimator.as_deref(),
                        Some("bayesian.basis.gcomp"),
                        "{}",
                        coordinate.text
                    );
                    let effect = result.estimate().as_effect().unwrap();
                    let cate = effect.heterogeneity.cate.as_ref().expect("row conditional effects");
                    let table_rows = rows();
                    assert_eq!(cate.len(), table_rows.len(), "{}", coordinate.text);
                    for (value, row) in cate.iter().zip(&table_rows) {
                        assert!(
                            (value - (2.0 + row.z)).abs() < DRAW_MEAN_TOLERANCE,
                            "{}: cate {value} at z = {}",
                            coordinate.text,
                            row.z
                        );
                    }
                }
                match coordinate.suite {
                    antecedent::RefuteSuite::None => {
                        assert!(result.refutations.is_empty(), "{}", coordinate.text);
                    }
                    _ => assert!(
                        !result.refutations.is_empty() || !result.predictive_checks.is_empty(),
                        "{}: validation",
                        coordinate.text
                    ),
                }
            },
        );
    });
}

// ------------------------------------------------------------------ bayesian.randomized_quantile

const CONTROL: [f64; 12] = [0.4, 0.8, 1.1, 1.3, 1.7, 2.2, 2.4, 2.9, 3.3, 3.6, 4.1, 4.6];
const ACTIVE: [f64; 12] = [0.9, 1.4, 1.8, 2.1, 2.6, 2.8, 3.1, 3.4, 3.9, 4.3, 4.4, 4.8];
const GRID: [f64; 6] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
const CONCENTRATION: f64 = 0.5;

/// Columns `t, y, w` of the randomized quantile design; `w` alternates and is independent of
/// both `t` and `y`.
fn quantile_table() -> TabularData {
    let mut outcome = CONTROL.to_vec();
    outcome.extend(ACTIVE);
    let mut treatment = vec![0.0; CONTROL.len()];
    treatment.extend(vec![1.0; ACTIVE.len()]);
    let w: Vec<f64> = (0..outcome.len()).map(|i| f64::from(u8::try_from(i % 2).unwrap())).collect();
    TabularData::from_f64_columns([
        ("t", treatment.as_slice()),
        ("y", outcome.as_slice()),
        ("w", w.as_slice()),
    ])
    .unwrap()
}

/// `t, y, w, m`: the same outcomes with two independent alternating columns for the tiers.
fn quantile_unknown_table() -> TabularData {
    let mut outcome = CONTROL.to_vec();
    outcome.extend(ACTIVE);
    let mut treatment = vec![0.0; CONTROL.len()];
    treatment.extend(vec![1.0; ACTIVE.len()]);
    let w: Vec<f64> = (0..outcome.len()).map(|i| f64::from(u8::try_from(i % 2).unwrap())).collect();
    let m: Vec<f64> =
        (0..outcome.len()).map(|i| f64::from(u8::try_from((i / 2) % 2).unwrap())).collect();
    TabularData::from_f64_columns([
        ("t", treatment.as_slice()),
        ("y", outcome.as_slice()),
        ("w", w.as_slice()),
        ("m", m.as_slice()),
    ])
    .unwrap()
}

/// Posterior mean of the CDF at each grid point under a symmetric Dirichlet prior on the
/// `grid.len() + 1` bins `(-inf, g0), [g0, g1), ..., [g_last, inf)`:
/// `(concentration + count(y < g_k's bins so far)) / (concentration * bins + n)`.
fn dirichlet_cdf_mean(values: &[f64]) -> Vec<f64> {
    let bins = GRID.len() + 1;
    let total = CONCENTRATION * bins as f64 + values.len() as f64;
    GRID.iter()
        .enumerate()
        .map(|(k, &threshold)| {
            let below = values.iter().filter(|&&v| v < threshold).count() as f64;
            (CONCENTRATION * (k + 1) as f64 + below) / total
        })
        .collect()
}

#[test]
fn bayesian_randomized_quantile_posterior_cdf_matches_the_dirichlet_closed_form() {
    // Hand check of the closed form against the counted bins of the committed reference
    // (control 0,2,3,3,2,2,0; treated 0,1,2,3,3,3,0; total mass 15.5).
    for (got, want) in
        dirichlet_cdf_mean(&CONTROL).iter().zip([0.5, 3.0, 6.5, 10.0, 12.5, 15.0].map(|a| a / 15.5))
    {
        assert!((got - want).abs() < 1e-12, "control {got} vs {want}");
    }
    for (got, want) in
        dirichlet_cdf_mean(&ACTIVE).iter().zip([0.5, 2.0, 4.5, 8.0, 11.5, 15.0].map(|a| a / 15.5))
    {
        assert!((got - want).abs() < 1e-12, "treated {got} vs {want}");
    }
    let control_cdf = dirichlet_cdf_mean(&CONTROL);
    let treated_cdf = dirichlet_cdf_mean(&ACTIVE);
    on_large_stack(move || {
        let design = Design {
            main: quantile_table(),
            unknown: quantile_unknown_table(),
            n_vars: 3,
            atom_a: vec![(0, 1)],
            atom_b: vec![(0, 1), (2, 1)],
            codetermined_tiers: vec![vec!["w"], vec!["t"], vec!["y"]],
            unknown_tiers: vec![vec!["w"], vec!["t", "m"], vec!["y"]],
        };
        let spec = EstimatorSpec::Default(EstimatorId::BayesianRandomizedQuantile);
        let mut config = BayesianConfig::conjugate().n_draws(4_000);
        config = config.design_model(DesignPosteriorModel::QuantileGrid {
            thresholds: Arc::from(GRID),
            bin_concentration: CONCENTRATION,
        });
        let inference = InferenceMode::Bayesian(config);
        for_each_matching(
            "bayesian.randomized_quantile|AverageEffect",
            |c| {
                c.graph == GraphKind::Dag
                    && c.structure != Structure::GraphPosterior
                    && c.suite_label == "none"
            },
            |coordinate| {
                let mut query = AverageEffectQuery::binary_ate(
                    VariableId::from_raw(0),
                    VariableId::from_raw(1),
                );
                query.outcome_functional = OutcomeFunctional::quantile(0.5);
                let result = run_query(
                    coordinate,
                    &spec,
                    &inference,
                    &design,
                    CausalQuery::AverageEffect(query),
                    None,
                    9,
                );
                if coordinate.structure != Structure::GraphPosterior {
                    assert_eq!(
                        result.logical_plan.estimator.as_deref(),
                        Some("bayesian.randomized_quantile"),
                        "{}",
                        coordinate.text
                    );
                }
                assert!(
                    coordinate.graph != GraphKind::Unknown || result.posterior_row().is_some(),
                    "{}: posterior row",
                    coordinate.text
                );
                let row = result
                    .posterior_row()
                    .unwrap_or_else(|| panic!("{}: no posterior row", coordinate.text));
                for (got, want) in row.vectors["control_cdf_mean"].iter().zip(&control_cdf) {
                    assert!(
                        (got - want).abs() < 1e-12,
                        "{}: control cdf {got} vs {want}",
                        coordinate.text
                    );
                }
                for (got, want) in row.vectors["treated_cdf_mean"].iter().zip(&treated_cdf) {
                    assert!(
                        (got - want).abs() < 1e-12,
                        "{}: treated cdf {got} vs {want}",
                        coordinate.text
                    );
                }
                match coordinate.suite {
                    antecedent::RefuteSuite::None => {
                        assert!(result.refutations.is_empty(), "{}", coordinate.text);
                    }
                    _ => assert!(!result.refutations.is_empty(), "{}: refuters", coordinate.text),
                }
            },
        );
    });
}
