//! Declared treatment levels of the propensity estimators and multi-level observational
//! contrasts.
//!
//! * Binary recoding: the query's active level becomes the 1 arm and its control level the 0
//!   arm, so `0/1`, `1/2` (active 2, control 1) and reversed codings estimate the same
//!   contrast, up to sign when active and control swap.
//! * Rows at neither declared level are refused unless the exclusion is declared, and a
//!   declared exclusion is counted and named on the result.
//! * Multi-level treatment: the pairwise-subpopulation and full-population
//!   generalized-propensity targets are checked against truths computed independently of the
//!   estimators (a closed form for the full population, numerical integration over the
//!   covariate law for the pairwise subpopulation), and they differ when the effect varies
//!   with the covariate.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_possible_truncation,
    clippy::float_cmp,
    clippy::needless_range_loop,
    reason = "test fixtures index small literals; level codes are exact labels"
)]

use std::sync::Arc;

use antecedent_core::{
    AssumptionSet, AverageEffectQuery, CausalSchemaBuilder, ExecutionContext, MeasurementSpec,
    RoleHint, SmallRoleSet, StreamDomain, ValueType, VariableId,
};
use antecedent_data::{
    Float64Column, OwnedColumn, OwnedColumnarStorage, TabularData, ValidityBitmap,
};
use antecedent_estimate::{
    AipwAte, AipwWorkspace, DmlAte, EffectEstimate, MultiLevelAipw, MultiLevelPopulation,
    MultiLevelTarget, OtherLevels, PropensityEstimationWorkspace, PropensityMatching,
    PropensityWeighting,
};
use antecedent_expr::{ExprId, IdentifiedEstimand};
use antecedent_kernels::{sigmoid, standard_normal};

#[path = "common/checks.rs"]
mod checks;

const T: VariableId = VariableId::from_raw(0);
const Y: VariableId = VariableId::from_raw(1);
const Z: VariableId = VariableId::from_raw(2);

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(11)
}

fn dataset(t: &[f64], y: &[f64], z: &[f64]) -> (TabularData, IdentifiedEstimand) {
    let n = t.len();
    let mut builder = CausalSchemaBuilder::new();
    for (name, hint) in [
        ("t", RoleHint::TreatmentCandidate),
        ("y", RoleHint::OutcomeCandidate),
        ("z", RoleHint::Context),
    ] {
        builder
            .add_variable(
                name,
                ValueType::Continuous,
                SmallRoleSet::from_hint(hint),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
    }
    let columns = [(T, t), (Y, y), (Z, z)]
        .into_iter()
        .map(|(id, values)| {
            OwnedColumn::Float64(
                Float64Column::new(id, Arc::from(values), ValidityBitmap::all_valid(n)).unwrap(),
            )
        })
        .collect();
    let storage =
        OwnedColumnarStorage::try_new(builder.build().unwrap(), columns, None, None).unwrap();
    let estimand =
        IdentifiedEstimand::backdoor("backdoor.adjustment", Arc::from([Z]), ExprId::from_raw(0));
    (TabularData::new(storage), estimand)
}

/// Confounded binary treatment: `P(T=1|z) = expit(-0.5 + z)`, `Y = 2 T + z + 0.5 noise`.
fn binary_columns(n: usize, seed: u64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut rng = ExecutionContext::for_tests(seed).rng.stream_for(StreamDomain::Estimate, 0xB1);
    let (mut t, mut y, mut z) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        let zi = standard_normal(&mut rng);
        let ti = f64::from(rng.next_f64() < sigmoid(-0.5 + zi));
        z[i] = zi;
        t[i] = ti;
        y[i] = 2.0 * ti + zi + 0.5 * standard_normal(&mut rng);
    }
    (t, y, z)
}

/// Level intercepts and covariate slopes of the three-level treatment: `E[Y(k) | z] =
/// ALPHA[k] + BETA[k] z`, so the effect of level 2 against level 0 is `3 + z`.
const ALPHA: [f64; 3] = [0.0, 1.0, 3.0];
const BETA: [f64; 3] = [1.0, 1.0, 2.0];
/// Multinomial-logit intercept and slope of levels 1 and 2 against level 0.
const LOGIT_1: (f64, f64) = (0.4, 0.8);
const LOGIT_2: (f64, f64) = (-0.3, -0.6);

fn three_level_columns(n: usize, seed: u64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut rng = ExecutionContext::for_tests(seed).rng.stream_for(StreamDomain::Estimate, 0xC3);
    let (mut t, mut y, mut z) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        let zi = standard_normal(&mut rng);
        let weights = [1.0, (LOGIT_1.0 + LOGIT_1.1 * zi).exp(), (LOGIT_2.0 + LOGIT_2.1 * zi).exp()];
        let mass: f64 = weights.iter().sum();
        let u = rng.next_f64() * mass;
        let level = if u < weights[0] {
            0
        } else if u < weights[0] + weights[1] {
            1
        } else {
            2
        };
        z[i] = zi;
        t[i] = level as f64;
        y[i] = ALPHA[level] + BETA[level] * zi + standard_normal(&mut rng);
    }
    (t, y, z)
}

/// `E[Y(2) - Y(0) | T in {0, 2}]` by a Riemann sum over the standard-normal
/// covariate: the effect `3 + z` averaged with the density of `z` among rows at level 0 or 2.
fn pairwise_truth() -> f64 {
    let step = 0.001;
    let (mut numerator, mut denominator) = (0.0_f64, 0.0_f64);
    let mut z = -9.0_f64;
    while z <= 9.0 {
        let density = (-z * z / 2.0).exp();
        let weights = [1.0, (LOGIT_1.0 + LOGIT_1.1 * z).exp(), (LOGIT_2.0 + LOGIT_2.1 * z).exp()];
        let at_pair = (weights[0] + weights[2]) / weights.iter().sum::<f64>();
        numerator += (ALPHA[2] - ALPHA[0] + (BETA[2] - BETA[0]) * z) * at_pair * density;
        denominator += at_pair * density;
        z += step;
    }
    numerator / denominator
}

/// `E[Y(2) - Y(0)]` over the whole population: `z` has mean zero.
const FULL_POPULATION_TRUTH: f64 = ALPHA[2] - ALPHA[0];

fn aipw() -> AipwAte {
    AipwAte::new().with_bootstrap_replicates(0)
}

fn fit_aipw(
    estimator: &AipwAte,
    data: &TabularData,
    estimand: &IdentifiedEstimand,
    query: &AverageEffectQuery,
) -> EffectEstimate {
    let problem = estimator.prepare(data, estimand, query).unwrap();
    estimator.fit(&problem, &mut AipwWorkspace::default(), &ctx(), AssumptionSet::new()).unwrap()
}

fn fit_weighting(
    data: &TabularData,
    estimand: &IdentifiedEstimand,
    query: &AverageEffectQuery,
) -> EffectEstimate {
    let estimator = PropensityWeighting::new().with_bootstrap_replicates(0);
    let problem = estimator.prepare(data, estimand, query).unwrap();
    estimator
        .fit(&problem, &mut PropensityEstimationWorkspace::default(), &ctx(), AssumptionSet::new())
        .unwrap()
}

fn fit_dml(
    data: &TabularData,
    estimand: &IdentifiedEstimand,
    query: &AverageEffectQuery,
) -> EffectEstimate {
    let estimator = DmlAte::new();
    let problem = estimator.prepare(data, estimand, query).unwrap();
    estimator.fit(&problem, &ctx(), AssumptionSet::new()).unwrap()
}

fn recode(t: &[f64], map: impl Fn(f64) -> f64) -> Vec<f64> {
    t.iter().map(|&v| map(v)).collect()
}

#[test]
fn a_shifted_coding_with_declared_levels_gives_the_identical_estimate() {
    let (t, y, z) = binary_columns(1500, 3);
    let (data, estimand) = dataset(&t, &y, &z);
    let base_query = AverageEffectQuery::with_levels(T, Y, 0.0, 1.0);
    let (shifted, _) = dataset(&recode(&t, |v| v + 1.0), &y, &z);
    // Treatment coded 1/2: control is the level 1, active the level 2.
    let shifted_query = AverageEffectQuery::with_levels(T, Y, 1.0, 2.0);

    let aipw_base = fit_aipw(&aipw(), &data, &estimand, &base_query);
    let aipw_shifted = fit_aipw(&aipw(), &shifted, &estimand, &shifted_query);
    assert_eq!(aipw_base.ate.to_bits(), aipw_shifted.ate.to_bits());
    assert_eq!(aipw_base.se_analytic().to_bits(), aipw_shifted.se_analytic().to_bits());

    let weighting_base = fit_weighting(&data, &estimand, &base_query);
    let weighting_shifted = fit_weighting(&shifted, &estimand, &shifted_query);
    assert_eq!(weighting_base.ate.to_bits(), weighting_shifted.ate.to_bits());

    let dml_base = fit_dml(&data, &estimand, &base_query);
    let dml_shifted = fit_dml(&shifted, &estimand, &shifted_query);
    assert_eq!(dml_base.ate.to_bits(), dml_shifted.ate.to_bits());

    // The estimate is the contrast between the levels, not scaled by their distance, and
    // lands on the simulated truth of 2.
    assert!((aipw_shifted.ate - 2.0).abs() < 0.3, "ate={}", aipw_shifted.ate);
}

#[test]
fn a_column_coded_the_other_way_round_gives_the_same_contrast() {
    let (t, y, z) = binary_columns(1500, 4);
    let (data, estimand) = dataset(&t, &y, &z);
    let base_query = AverageEffectQuery::with_levels(T, Y, 0.0, 1.0);
    // The same units, treatment coded 1 for the controls and 0 for the treated: the declared
    // active level 0 is the group the original coding called treated.
    let (flipped, _) = dataset(&recode(&t, |v| 1.0 - v), &y, &z);
    let flipped_query = AverageEffectQuery::with_levels(T, Y, 1.0, 0.0);

    let base = fit_weighting(&data, &estimand, &base_query);
    let same_units = fit_weighting(&flipped, &estimand, &flipped_query);
    assert!((base.ate - same_units.ate).abs() < 1e-9, "{} vs {}", base.ate, same_units.ate);

    let base_aipw = fit_aipw(&aipw(), &data, &estimand, &base_query);
    let same_units_aipw = fit_aipw(&aipw(), &flipped, &estimand, &flipped_query);
    // Folds are dealt on the column's own level order, which this coding reverses, so the
    // cross-fitted fits agree only up to fold-plan noise.
    assert!((base_aipw.ate - same_units_aipw.ate).abs() < 3.0 * base_aipw.se_analytic());
}

#[test]
fn swapping_active_and_control_flips_the_sign() {
    let (t, y, z) = binary_columns(1500, 5);
    let (data, estimand) = dataset(&t, &y, &z);
    let forward = AverageEffectQuery::with_levels(T, Y, 0.0, 1.0);
    let reversed = AverageEffectQuery::with_levels(T, Y, 1.0, 0.0);

    let weighting = fit_weighting(&data, &estimand, &forward);
    let weighting_reversed = fit_weighting(&data, &estimand, &reversed);
    assert!((weighting.ate + weighting_reversed.ate).abs() < 1e-9);

    // The folds are dealt on the original levels, so swapping the arms deals the same folds
    // and the contrasts are exact negatives.
    let estimate = fit_aipw(&aipw(), &data, &estimand, &forward);
    let estimate_reversed = fit_aipw(&aipw(), &data, &estimand, &reversed);
    assert!(estimate.ate > 1.0);
    assert!(
        (estimate.ate + estimate_reversed.ate).abs() < 1e-9,
        "{} vs {}",
        estimate.ate,
        estimate_reversed.ate
    );
}

#[test]
fn rows_at_undeclared_levels_are_refused_unless_the_exclusion_is_declared() {
    let (t, y, z) = three_level_columns(1200, 6);
    let (data, estimand) = dataset(&t, &y, &z);
    let query = AverageEffectQuery::with_levels(T, Y, 0.0, 2.0);
    let middle = t.iter().filter(|&&v| v == 1.0).count() as u64;
    assert!(middle > 0);

    for refusal in [
        aipw().prepare(&data, &estimand, &query).unwrap_err(),
        PropensityWeighting::new().prepare(&data, &estimand, &query).unwrap_err(),
        DmlAte::new().prepare(&data, &estimand, &query).unwrap_err(),
    ] {
        let (code, message) = checks::coded_refusal(refusal);
        assert_eq!(code, "treatment_level_undeclared");
        assert!(message.starts_with("treatment_levels.undeclared_level:"), "{message}");
        assert!(message.contains(&format!("{middle} complete-case rows")), "{message}");
    }
}

#[test]
fn a_declared_exclusion_is_counted_and_names_its_population() {
    let (t, y, z) = three_level_columns(1200, 7);
    let (data, estimand) = dataset(&t, &y, &z);
    let query = AverageEffectQuery::with_levels(T, Y, 0.0, 2.0);
    let count = |level: f64| t.iter().filter(|&&v| v == level).count() as u64;

    let estimator = aipw().with_other_levels(OtherLevels::ExcludeUndeclared);
    let problem = estimator.prepare(&data, &estimand, &query).unwrap();
    assert_eq!(problem.nrows as u64, count(0.0) + count(2.0));
    // The retained rows are exactly the rows at the two declared levels, recoded to 0/1.
    let expected: Vec<u64> =
        (0..t.len()).filter(|&i| t[i] != 1.0).map(|i| u64::try_from(i).unwrap()).collect();
    let got: Vec<u64> = problem.row_index.iter().map(|id| id.get()).collect();
    assert_eq!(got, expected);
    assert!(problem.treatment.iter().all(|&v| v == 0.0 || v == 1.0));

    let effect = estimator
        .fit(&problem, &mut AipwWorkspace::default(), &ctx(), AssumptionSet::new())
        .unwrap();
    let levels = effect.diagnostics.treatment_levels.expect("levels are reported");
    assert_eq!((levels.active, levels.control), (2.0, 0.0));
    assert_eq!(levels.rows_at_active, count(2.0));
    assert_eq!(levels.rows_at_control, count(0.0));
    assert_eq!(levels.excluded_rows, count(1.0));
    assert_eq!(levels.other_levels, OtherLevels::ExcludeUndeclared);
    assert_eq!(levels.restricted_population(), Some("rows at the two declared levels"));
}

#[test]
fn a_fully_coded_column_reports_no_exclusion() {
    let (t, y, z) = binary_columns(600, 8);
    let (data, estimand) = dataset(&t, &y, &z);
    let query = AverageEffectQuery::with_levels(T, Y, 0.0, 1.0);
    let effect = fit_aipw(&aipw(), &data, &estimand, &query);
    let levels = effect.diagnostics.treatment_levels.expect("levels are reported");
    assert_eq!(levels.excluded_rows, 0);
    assert_eq!(levels.restricted_population(), None);
    assert_eq!(levels.rows_used(), 600);
}

#[test]
fn the_full_population_and_pairwise_targets_recover_their_own_truths() {
    let (t, y, z) = three_level_columns(9000, 21);
    let (data, estimand) = dataset(&t, &y, &z);
    let query = AverageEffectQuery::with_levels(T, Y, 0.0, 2.0);

    let full = MultiLevelAipw::full_population(vec![0.0, 1.0, 2.0])
        .fit(&data, &estimand, &query, &ctx(), AssumptionSet::new())
        .unwrap();
    let pairwise = MultiLevelAipw::pairwise_subpopulation()
        .fit(&data, &estimand, &query, &ctx(), AssumptionSet::new())
        .unwrap();

    // Semantics and populations are stated on the result, never inferred.
    assert!(matches!(full.target, MultiLevelTarget::FullPopulation { .. }));
    assert_eq!(full.target_population, MultiLevelPopulation::AllCompleteCaseRows);
    assert_eq!(full.excluded_rows, 0);
    assert_eq!(pairwise.target, MultiLevelTarget::PairwiseSubpopulation);
    assert_eq!(pairwise.target_population, MultiLevelPopulation::RowsAtTheTwoDeclaredLevels);
    assert_eq!(pairwise.excluded_rows, t.iter().filter(|&&v| v == 1.0).count() as u64);

    let pairwise_truth = pairwise_truth();
    let (full_se, pairwise_se) = (full.effect.se_analytic(), pairwise.effect.se_analytic());
    assert!(full_se > 0.0 && full_se < 0.15, "se={full_se}");
    assert!(
        (full.effect.ate - FULL_POPULATION_TRUTH).abs() < 4.0 * full_se,
        "full ate={} truth={FULL_POPULATION_TRUTH} se={full_se}",
        full.effect.ate
    );
    assert!(
        (pairwise.effect.ate - pairwise_truth).abs() < 4.0 * pairwise_se,
        "pairwise ate={} truth={pairwise_truth} se={pairwise_se}",
        pairwise.effect.ate
    );

    // The effect varies with the covariate, so the two targets are different numbers: the
    // truths differ by about 0.4 and the estimates resolve it.
    assert!((FULL_POPULATION_TRUTH - pairwise_truth).abs() > 0.3);
    assert!(
        full.effect.ate - pairwise.effect.ate > 0.1,
        "full={} pairwise={}",
        full.effect.ate,
        pairwise.effect.ate
    );

    // Every level's standardized mean is its own whole-population mean outcome.
    assert_eq!(full.level_means.len(), 3);
    for (mean, alpha) in full.level_means.iter().zip(ALPHA) {
        assert!(mean.se > 0.0, "{mean:?}");
        assert!((mean.mean - alpha).abs() < 4.0 * mean.se, "{mean:?} truth={alpha}");
        assert!(mean.propensity_range.0 > 0.0 && mean.propensity_range.1 < 1.0);
    }
}

#[test]
fn full_population_standardization_refuses_rows_at_an_undeclared_level() {
    let (t, y, z) = three_level_columns(900, 22);
    let (data, estimand) = dataset(&t, &y, &z);
    let query = AverageEffectQuery::with_levels(T, Y, 0.0, 2.0);
    let error = MultiLevelAipw::full_population(vec![0.0, 2.0])
        .fit(&data, &estimand, &query, &ctx(), AssumptionSet::new())
        .unwrap_err();
    let (code, message) = checks::coded_refusal(error);
    assert_eq!(code, "treatment_level_undeclared");
    assert!(message.starts_with("treatment_levels.full_population_undeclared_level:"), "{message}");
}

#[test]
fn the_contrast_levels_must_be_in_the_declared_set() {
    let (t, y, z) = three_level_columns(900, 23);
    let (data, estimand) = dataset(&t, &y, &z);
    let query = AverageEffectQuery::with_levels(T, Y, 0.0, 3.0);
    let error = MultiLevelAipw::full_population(vec![0.0, 1.0, 2.0])
        .fit(&data, &estimand, &query, &ctx(), AssumptionSet::new())
        .unwrap_err();
    let (code, message) = checks::coded_refusal(error);
    assert_eq!(code, "invalid_argument");
    assert!(message.starts_with("treatment_levels.contrast_level_not_declared:"), "{message}");
}

#[test]
fn a_consumer_replays_the_contrast_and_the_level_means_from_the_score_table() {
    let (t, y, z) = three_level_columns(3000, 24);
    let (data, estimand) = dataset(&t, &y, &z);
    let query = AverageEffectQuery::with_levels(T, Y, 0.0, 2.0);
    let full = MultiLevelAipw::full_population(vec![0.0, 1.0, 2.0])
        .fit(&data, &estimand, &query, &ctx(), AssumptionSet::new())
        .unwrap();
    let table = full.effect.scores.table.as_ref().expect("score table");
    assert_eq!(table.n_columns(), 3);
    assert_eq!(table.nuisance_provenance.as_ref(), "gps.crossfit.multinomial_logit.ols.v1");
    let n = table.n_rows as f64;
    // The means are the plain column means of the per-level scores, and the contrast is the
    // difference of two of them: one influence-function path for both.
    for (index, mean) in full.level_means.iter().enumerate() {
        let column_mean = table.column(index).unwrap().iter().sum::<f64>() / n;
        assert!((column_mean - mean.mean).abs() < 1e-9, "level {index}");
    }
    let replayed = full.level_means[2].mean - full.level_means[0].mean;
    assert!((replayed - full.effect.ate).abs() < 1e-9);
    // The contrast's variance is the influence variance of the score difference.
    let difference: Vec<f64> =
        table.column(2).unwrap().iter().zip(table.column(0).unwrap()).map(|(a, b)| a - b).collect();
    let mean = difference.iter().sum::<f64>() / n;
    let variance = difference.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let replayed_se = (variance / n).sqrt();
    assert!(
        (replayed_se - full.effect.se_analytic()).abs() < 0.02 * replayed_se,
        "{replayed_se} vs {}",
        full.effect.se_analytic()
    );
}

// --- Acceptance datasets -----------------------------------------------------------------------

/// Effect of the treatment in the confounded binary acceptance dataset.
const BINARY_EFFECT: f64 = 0.5;

/// Moderately confounded binary treatment of the acceptance cases:
/// `P(T=1|z) = expit(-0.2 + 0.8 z)` and `Y = 0.5 T + 0.7 z + 0.5 noise`.
fn confounded_binary(n: usize, seed: u64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut rng = ExecutionContext::for_tests(seed).rng.stream_for(StreamDomain::Estimate, 0xB2);
    let (mut t, mut y, mut z) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        let zi = standard_normal(&mut rng);
        let ti = f64::from(rng.next_f64() < sigmoid(-0.2 + 0.8 * zi));
        z[i] = zi;
        t[i] = ti;
        y[i] = BINARY_EFFECT * ti + 0.7 * zi + 0.5 * standard_normal(&mut rng);
    }
    (t, y, z)
}

/// Mean outcome of each level of the confounded three-level acceptance dataset.
const LEVEL_EFFECTS: [f64; 3] = [0.0, 0.4, 0.9];
/// Multinomial-logit intercept and slope of levels 1 and 2 against level 0.
const CONFOUNDED_LOGIT: [(f64, f64); 2] = [(0.3, 0.6), (-0.2, -0.5)];
/// Slope of the outcome on the covariate, common to the three levels so the effect of every
/// pair of levels is constant in the covariate.
const OUTCOME_SLOPE: f64 = 0.6;

/// Three-level treatment drawn from a multinomial logit of the covariate, with level means
/// `LEVEL_EFFECTS`: `E[Y(k) | z] = LEVEL_EFFECTS[k] + 0.6 z`, so the effect of level `a` against
/// level `b` is `LEVEL_EFFECTS[a] - LEVEL_EFFECTS[b]` over every population.
fn confounded_three_level(n: usize, seed: u64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut rng = ExecutionContext::for_tests(seed).rng.stream_for(StreamDomain::Estimate, 0xC4);
    let (mut t, mut y, mut z) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        let zi = standard_normal(&mut rng);
        let weights = [
            1.0,
            (CONFOUNDED_LOGIT[0].0 + CONFOUNDED_LOGIT[0].1 * zi).exp(),
            (CONFOUNDED_LOGIT[1].0 + CONFOUNDED_LOGIT[1].1 * zi).exp(),
        ];
        let u = rng.next_f64() * weights.iter().sum::<f64>();
        let level = if u < weights[0] {
            0
        } else if u < weights[0] + weights[1] {
            1
        } else {
            2
        };
        z[i] = zi;
        t[i] = level as f64;
        y[i] = LEVEL_EFFECTS[level] + OUTCOME_SLOPE * zi + 0.5 * standard_normal(&mut rng);
    }
    (t, y, z)
}

fn fit_matching(
    data: &TabularData,
    estimand: &IdentifiedEstimand,
    query: &AverageEffectQuery,
) -> EffectEstimate {
    let estimator = PropensityMatching::new();
    let problem = estimator.prepare(data, estimand, query).unwrap();
    estimator
        .fit(&problem, &mut PropensityEstimationWorkspace::default(), &ctx(), AssumptionSet::new())
        .unwrap()
}

/// Rows at a level of the column.
fn rows_at(t: &[f64], level: f64) -> u64 {
    t.iter().filter(|&&v| v == level).count() as u64
}

/// The declared contrast of a fit as its result states it: active and control level and the
/// rows each arm and the exclusion hold.
fn stated_contrast(effect: &EffectEstimate) -> (f64, f64, u64, u64, u64) {
    let levels = effect.diagnostics.treatment_levels.expect("the fit states its declared levels");
    (
        levels.active,
        levels.control,
        levels.rows_at_active,
        levels.rows_at_control,
        levels.excluded_rows,
    )
}

/// Fit the same units coded 0/1 and coded 1/2 with active 2 and control 1: the estimate is
/// bit-identical to the 0/1 estimate, and the result states the declared levels.
fn assert_declared_two_over_one_matches_zero_one(
    seed: u64,
    fit: impl Fn(&TabularData, &IdentifiedEstimand, &AverageEffectQuery) -> EffectEstimate,
) {
    let (t, y, z) = confounded_binary(2000, seed);
    let (zero_one, estimand) = dataset(&t, &y, &z);
    let (one_two, _) = dataset(&recode(&t, |v| v + 1.0), &y, &z);
    let base = fit(&zero_one, &estimand, &AverageEffectQuery::with_levels(T, Y, 0.0, 1.0));
    let declared = fit(&one_two, &estimand, &AverageEffectQuery::with_levels(T, Y, 1.0, 2.0));

    // The recoding changes nothing about the units, so the estimate is the same number.
    assert_eq!(base.ate.to_bits(), declared.ate.to_bits(), "{} vs {}", base.ate, declared.ate);
    // It is the contrast of the arms, not scaled by the distance between the labels, and it
    // lands on the simulated effect.
    assert!((declared.ate - BINARY_EFFECT).abs() < 0.25, "ate={}", declared.ate);

    // The estimand states the declared levels and the arms they selected.
    let (active, control, at_active, at_control, excluded) = stated_contrast(&declared);
    assert_eq!((active, control), (2.0, 1.0));
    assert_eq!((at_active, at_control, excluded), (rows_at(&t, 1.0), rows_at(&t, 0.0), 0));
    assert_eq!(declared.diagnostics.treatment_levels.unwrap().restricted_population(), None);
    let reference = stated_contrast(&base);
    assert_eq!((reference.0, reference.1), (1.0, 0.0));
    assert_eq!((reference.2, reference.3, reference.4), (at_active, at_control, excluded));
}

#[test]
fn propensity_weighting_honors_declared_active_two_control_one() {
    assert_declared_two_over_one_matches_zero_one(2_307_708, fit_weighting);
}

#[test]
fn aipw_honors_declared_active_two_control_one() {
    assert_declared_two_over_one_matches_zero_one(8_932_728, |data, estimand, query| {
        let estimate = fit_aipw(&aipw(), data, estimand, query);
        assert!(estimate.se_analytic() > 0.0);
        estimate
    });
}

#[test]
fn matching_honors_declared_active_two_control_one() {
    assert_declared_two_over_one_matches_zero_one(8_959_944, fit_matching);
}

#[test]
fn reversing_the_declared_contrast_negates_the_aipw_estimate() {
    let (t, y, z) = confounded_binary(2000, 9_609_035);
    // The column is coded 1/2: 2 is the treated arm.
    let (data, estimand) = dataset(&recode(&t, |v| v + 1.0), &y, &z);
    let forward =
        fit_aipw(&aipw(), &data, &estimand, &AverageEffectQuery::with_levels(T, Y, 1.0, 2.0));
    let reversed =
        fit_aipw(&aipw(), &data, &estimand, &AverageEffectQuery::with_levels(T, Y, 2.0, 1.0));
    assert!((forward.ate - BINARY_EFFECT).abs() < 0.25, "ate={}", forward.ate);
    // The cross-fit folds are dealt on the column's own levels, so swapping the arms refits
    // the same models and the contrasts are exact negatives, as are the influence-function
    // standard errors' squares.
    assert!(
        (forward.ate + reversed.ate).abs() < 1e-9,
        "2-versus-1 {} and 1-versus-2 {}",
        forward.ate,
        reversed.ate
    );
    assert!((forward.se_analytic() - reversed.se_analytic()).abs() < 1e-9);
    let (active, control, ..) = stated_contrast(&reversed);
    assert_eq!((active, control), (1.0, 2.0));
}

#[test]
fn equivalent_treatment_recodings_agree() {
    let (t, y, z) = confounded_binary(2000, 7_866_192);
    let (base, estimand) = dataset(&t, &y, &z);
    // Control first, active second: 0/1, 1/2, 10/20 and the labels `a`/`b` carried as the
    // arbitrary codes a caller assigns them (`a` = -2.5, `b` = 7.25).
    let codings = [(0.0, 1.0), (1.0, 2.0), (10.0, 20.0), (-2.5, 7.25)];
    let fits = |data: &TabularData, query: &AverageEffectQuery| {
        [
            fit_aipw(&aipw(), data, &estimand, query),
            fit_weighting(data, &estimand, query),
            fit_matching(data, &estimand, query),
        ]
    };
    let reference = fits(&base, &AverageEffectQuery::with_levels(T, Y, 0.0, 1.0));
    for (control, active) in codings {
        let (data, _) = dataset(&recode(&t, |v| if v == 1.0 { active } else { control }), &y, &z);
        let recoded = fits(&data, &AverageEffectQuery::with_levels(T, Y, control, active));
        for (got, want) in recoded.iter().zip(&reference) {
            assert_eq!(got.ate.to_bits(), want.ate.to_bits(), "coding {control}/{active}");
            // The estimand in words is the same contrast over the same arms and rows; only
            // the labels it names differ, and they are the declared ones.
            let (a, c, at_active, at_control, excluded) = stated_contrast(got);
            let (_, _, ref_active, ref_control, ref_excluded) = stated_contrast(want);
            assert_eq!((a, c), (active, control));
            assert_eq!((at_active, at_control, excluded), (ref_active, ref_control, ref_excluded));
        }
    }
}

// --- Pairwise and full-population multi-level contrasts -----------------------------------------

/// Cross-fitted pairwise contrast of level 2 against level 1 on three-level data.
#[test]
fn multilevel_pairwise_population_exclusions_are_reported() {
    let (t, y, z) = confounded_three_level(3000, 5_088_953);
    let (data, estimand) = dataset(&t, &y, &z);
    let query = AverageEffectQuery::with_levels(T, Y, 1.0, 2.0);
    let pairwise = MultiLevelAipw::pairwise_subpopulation()
        .fit(&data, &estimand, &query, &ctx(), AssumptionSet::new())
        .unwrap();

    // The rows at the third level are excluded, counted, and the population is stated.
    let at_other = rows_at(&t, 0.0);
    assert!(at_other > 0);
    assert_eq!(pairwise.excluded_rows, at_other);
    assert_eq!(pairwise.target, MultiLevelTarget::PairwiseSubpopulation);
    assert_eq!(pairwise.target_population, MultiLevelPopulation::RowsAtTheTwoDeclaredLevels);
    assert_eq!(pairwise.target_population.as_str(), "rows at the two declared levels");
    let mut by_level: Vec<(f64, u64)> =
        pairwise.rows_by_level.iter().map(|r| (r.level, r.rows)).collect();
    by_level.sort_by(|a, b| a.0.total_cmp(&b.0));
    assert_eq!(by_level, vec![(1.0, rows_at(&t, 1.0)), (2.0, rows_at(&t, 2.0))]);
    let used = pairwise.effect.diagnostics.treatment_levels.expect("levels are reported");
    assert_eq!((used.active, used.control, used.excluded_rows), (2.0, 1.0, at_other));
    assert_eq!(used.restricted_population(), Some("rows at the two declared levels"));
    // The effect of level 2 against level 1 is constant in the covariate, so the pairwise
    // population does not move it away from the simulated 0.5.
    let truth = LEVEL_EFFECTS[2] - LEVEL_EFFECTS[1];
    let se = pairwise.effect.se_analytic();
    assert!(se > 0.0 && se < 0.1, "se={se}");
    assert!((pairwise.effect.ate - truth).abs() < 4.0 * se, "{} vs {truth}", pairwise.effect.ate);
}

/// Least-squares coefficients by the normal equations, solved with partial pivoting:
/// a computation of the oracle that shares no code with the estimators.
fn least_squares(design: &[Vec<f64>], y: &[f64]) -> Vec<f64> {
    let p = design[0].len();
    let mut a = vec![vec![0.0; p + 1]; p];
    for (row, &yi) in design.iter().zip(y) {
        for j in 0..p {
            for k in 0..p {
                a[j][k] += row[j] * row[k];
            }
            a[j][p] += row[j] * yi;
        }
    }
    for col in 0..p {
        let pivot = (col..p).max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs())).unwrap();
        a.swap(col, pivot);
        for row in col + 1..p {
            let factor = a[row][col] / a[col][col];
            for k in col..=p {
                a[row][k] -= factor * a[col][k];
            }
        }
    }
    let mut beta = vec![0.0; p];
    for col in (0..p).rev() {
        let tail: f64 = (col + 1..p).map(|k| a[col][k] * beta[k]).sum();
        beta[col] = (a[col][p] - tail) / a[col][col];
    }
    beta
}

/// Standardized mean of every level by g-computation on a linear outcome model with level
/// intercepts and a common covariate slope: the outcome model of the simulation, fitted by
/// least squares and averaged over every row's covariate.
fn standardized_means_by_regression(t: &[f64], y: &[f64], z: &[f64]) -> [f64; 3] {
    let design: Vec<Vec<f64>> = t
        .iter()
        .zip(z)
        .map(|(&level, &zi)| vec![1.0, f64::from(level == 1.0), f64::from(level == 2.0), zi])
        .collect();
    let beta = least_squares(&design, y);
    let mean_z = z.iter().sum::<f64>() / z.len() as f64;
    [
        beta[0] + beta[3] * mean_z,
        beta[0] + beta[1] + beta[3] * mean_z,
        beta[0] + beta[2] + beta[3] * mean_z,
    ]
}

#[test]
fn multilevel_generalized_propensity_standardizes_to_the_full_population() {
    let (t, y, z) = confounded_three_level(3000, 6_818_193);
    let (data, estimand) = dataset(&t, &y, &z);
    let query = AverageEffectQuery::with_levels(T, Y, 0.0, 2.0);
    let full = MultiLevelAipw::full_population(vec![0.0, 1.0, 2.0])
        .fit(&data, &estimand, &query, &ctx(), AssumptionSet::new())
        .unwrap();

    // The population is the full observed population, stated and never inferred.
    assert!(matches!(full.target, MultiLevelTarget::FullPopulation { .. }));
    assert_eq!(full.target_population, MultiLevelPopulation::AllCompleteCaseRows);
    assert_eq!(full.target_population.as_str(), "all complete-case rows");
    assert_eq!(full.excluded_rows, 0);
    assert_eq!(full.rows_by_level.iter().map(|r| r.rows).sum::<u64>(), 3000);

    // A standardized mean per declared level, each over the whole population.
    assert_eq!(full.level_means.len(), 3);
    let oracle = standardized_means_by_regression(&t, &y, &z);
    for ((mean, truth), regression) in full.level_means.iter().zip(LEVEL_EFFECTS).zip(oracle) {
        assert!(mean.se > 0.0 && mean.se < 0.1, "{mean:?}");
        assert!((mean.mean - truth).abs() < 4.0 * mean.se, "{mean:?} truth={truth}");
        assert!((mean.mean - regression).abs() < 4.0 * mean.se, "{mean:?} oracle={regression}");
    }

    // The contrast is the difference of the standardized means, and agrees with both the
    // simulated truth and the regression g-computation computed independently of it.
    let contrast = full.level_means[2].mean - full.level_means[0].mean;
    assert!((contrast - full.effect.ate).abs() < 1e-9);
    let (full_se, oracle_contrast) = (full.effect.se_analytic(), oracle[2] - oracle[0]);
    assert!(
        (full.effect.ate - (LEVEL_EFFECTS[2] - LEVEL_EFFECTS[0])).abs() < 4.0 * full_se,
        "ate={}",
        full.effect.ate
    );
    assert!(
        (full.effect.ate - oracle_contrast).abs() < 4.0 * full_se,
        "ate={} oracle={oracle_contrast}",
        full.effect.ate
    );
}

#[test]
fn an_undeclared_multilevel_contrast_is_refused_naming_the_observed_levels() {
    let (t, y, z) = confounded_three_level(600, 1_279_651);
    let (data, estimand) = dataset(&t, &y, &z);
    // Only two of the three levels are declared and no exclusion policy is: nothing reduces
    // the estimand to the pair silently.
    let query = AverageEffectQuery::with_levels(T, Y, 0.0, 2.0);
    let at = [rows_at(&t, 0.0), rows_at(&t, 1.0), rows_at(&t, 2.0)];
    for refusal in [
        aipw().prepare(&data, &estimand, &query).unwrap_err(),
        PropensityWeighting::new().prepare(&data, &estimand, &query).unwrap_err(),
        PropensityMatching::new().prepare(&data, &estimand, &query).unwrap_err(),
        DmlAte::new().prepare(&data, &estimand, &query).unwrap_err(),
    ] {
        let (code, message) = checks::coded_refusal(refusal);
        assert_eq!(code, "treatment_level_undeclared");
        assert!(message.starts_with("treatment_levels.undeclared_level:"), "{message}");
        let listing = format!(
            "observed treatment levels: 0 ({} rows), 1 ({} rows), 2 ({} rows)",
            at[0], at[1], at[2]
        );
        assert!(message.contains(&listing), "{message}");
    }
    // The full-population estimator refuses a level set that leaves a level out the same way.
    let error = MultiLevelAipw::full_population(vec![0.0, 2.0])
        .fit(&data, &estimand, &query, &ctx(), AssumptionSet::new())
        .unwrap_err();
    let (code, message) = checks::coded_refusal(error);
    assert_eq!(code, "treatment_level_undeclared");
    assert!(message.contains("observed treatment levels: 0 ("), "{message}");
}
