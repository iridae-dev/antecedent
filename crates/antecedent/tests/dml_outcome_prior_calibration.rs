//! Coverage of the cross-fitted AIPW interval under a prior on the outcome nuisance.
//!
//! A Gaussian coefficient prior regularizes each fold's per-arm outcome regression; the doubly
//! robust correction, the cross-fitting and the influence-function interval `estimate ± z · se`
//! keep their form. The propensity is correctly specified in every design, so the prior shapes
//! only the efficiency of the estimate: the interval must cover whatever the prior says. Three
//! prior strengths are measured on one design (effect 2, two confounders): a weak prior (the
//! near-unpenalized fit), an informative prior centred on the true outcome slopes, and a tight
//! prior centred on coefficients that are badly wrong.
//!
//! The designs are `#[ignore]`d measurement designs registered in
//! `parity/calibration_groups.toml`; the coverage record is the only missing piece. The
//! fixed-seed truth checks of the same estimator are
//! `crates/antecedent-estimate/tests/dml_outcome_prior_reference.rs`.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

mod common;

use std::sync::Arc;

use antecedent_core::{
    AssumptionSet, AverageEffectQuery, ExecutionContext, StreamDomain, VariableId,
};
use antecedent_data::TabularData;
use antecedent_estimate::{AipwAte, AipwWorkspace, DmlAte, HydrateMapping, OutcomePriorSource};
use antecedent_expr::{ExprId, IdentifiedEstimand};
use antecedent_kernels::standard_normal;
use antecedent_prob::PosteriorQuantityKind;
use common::calibration::{
    Construction, CoverageTally, REPORTED_LEVEL, RecordKey, ScopeFacts, Z95, grid_n, grid_seed,
    map_replicates, n_sim,
};

const INTERVAL: &str = "analytic_se";
const TRUTH: f64 = 2.0;
const DGP: &str = "confounded_linear_two_covariates";

/// One replicate: simulate `n` rows, fit the prior-regularized cross-fitted AIPW and return its
/// normal interval with the row count.
fn replicate(n: usize, seed: u64, mean: [f64; 3], sd: [f64; 3]) -> Option<((f64, f64), u64)> {
    replicate_on(Route::Dml, n, seed, mean, sd)
}

/// The estimator that carries the prior.
#[derive(Clone, Copy)]
enum Route {
    /// The cross-fitted DML AIPW score.
    Dml,
    /// The cross-fitted AIPW estimator with no bootstrap.
    Aipw,
    /// The DML route under a composed external prior with named coordinates (the treatment
    /// coordinate has no outcome-regression column and is dropped).
    DmlComposed,
    /// The AIPW estimator route under the same composed prior.
    AipwComposed,
}

/// The composed prior of a named design: coordinates `intercept`, `coef_z0`, `coef_z1` on the
/// conjugate scale `V0 = sd^2`, plus a treatment coordinate the outcome regression lacks.
fn composed_source(mean: [f64; 3], sd: [f64; 3]) -> Option<OutcomePriorSource> {
    let mut prior = antecedent_prob::PriorSet::new();
    prior.push(antecedent_prob::PriorSpec::GaussianCoefficients(
        antecedent_prob::GaussianCoefficientPrior {
            mean: Arc::from(vec![mean[0], mean[1], mean[2], 2.0]),
            variance: Arc::from(vec![sd[0] * sd[0], sd[1] * sd[1], sd[2] * sd[2], 1.0]),
        },
    ));
    let names: Vec<Arc<str>> =
        ["intercept", "coef_z0", "coef_z1", "coef_t"].iter().map(|n| Arc::from(*n)).collect();
    OutcomePriorSource::from_composed(Some(Arc::from(names)), prior, "composed:calibration").ok()
}

fn replicate_on(
    route: Route,
    n: usize,
    seed: u64,
    mean: [f64; 3],
    sd: [f64; 3],
) -> Option<((f64, f64), u64)> {
    let mut rng = ExecutionContext::for_tests(seed).rng.stream_for(StreamDomain::Test, 0x0FA1);
    let (mut t, mut y) = (vec![0.0; n], vec![0.0; n]);
    let (mut z0, mut z1) = (vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        z0[i] = standard_normal(&mut rng);
        z1[i] = standard_normal(&mut rng);
        let eta = 0.6 * z0[i] - 0.4 * z1[i];
        t[i] = f64::from(rng.next_f64() < 1.0 / (1.0 + (-eta).exp()));
        y[i] = TRUTH * t[i] + z0[i] + 0.5 * z1[i] + 0.5 * standard_normal(&mut rng);
    }
    let columns = [("t", t), ("y", y), ("z0", z0), ("z1", z1)];
    let borrowed: Vec<(&str, &[f64])> = columns.iter().map(|(n, v)| (*n, v.as_slice())).collect();
    let data = TabularData::from_f64_columns(borrowed).ok()?;
    let estimand = IdentifiedEstimand::backdoor(
        "backdoor.adjustment",
        Arc::from([VariableId::from_raw(2), VariableId::from_raw(3)]),
        ExprId::from_raw(0),
    );
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let names = ["intercept", "coef_z0", "coef_z1"];
    let mut quantities: Vec<PosteriorQuantityKind> = names
        .iter()
        .enumerate()
        .map(|(index, name)| PosteriorQuantityKind::Coefficient {
            index,
            name: Some(Arc::from(*name)),
        })
        .collect();
    quantities.push(PosteriorQuantityKind::ResidualVariance);
    let source = match route {
        Route::DmlComposed | Route::AipwComposed => composed_source(mean, sd)?,
        Route::Dml | Route::Aipw => OutcomePriorSource::new(
            quantities,
            vec![mean[0], mean[1], mean[2], 1.0],
            vec![sd[0], sd[1], sd[2], 0.0],
            None,
            HydrateMapping::IdenticalCoefficientSubspace,
            "blake3:calibration-source",
        )
        .ok()?,
    };
    let ctx = ExecutionContext::for_tests(seed);
    let fit = match route {
        Route::Dml | Route::DmlComposed => {
            let estimator = DmlAte::new().with_outcome_prior(source);
            let problem = estimator.prepare(&data, &estimand, &query).ok()?;
            estimator.fit(&problem, &ctx, AssumptionSet::new()).ok()?
        }
        Route::Aipw | Route::AipwComposed => {
            let estimator = AipwAte::new().with_bootstrap_replicates(0).with_outcome_prior(source);
            let problem = estimator.prepare(&data, &estimand, &query).ok()?;
            estimator
                .fit(&problem, &mut AipwWorkspace::default(), &ctx, AssumptionSet::new())
                .ok()?
        }
    };
    let se = fit.se_analytic();
    (fit.ate.is_finite() && se.is_finite() && se > 0.0)
        .then_some(((fit.ate - Z95 * se, fit.ate + Z95 * se), n as u64))
}

fn construction(estimator: &str) -> Construction {
    Construction {
        query: "AverageEffect".into(),
        graph_class: "Dag".into(),
        structure: "fixed".into(),
        modality: "tabular".into(),
        inference: "Frequentist".into(),
        estimator: estimator.into(),
        interval_method: INTERVAL.into(),
        se_kind: "homoskedastic".into(),
        dependence: "iid".into(),
        posterior: String::new(),
        functional: "all_observed.mean+outcome=gaussian_prior".into(),
        identification: "point".into(),
        reported_level: REPORTED_LEVEL,
        unit: None,
    }
}

fn measure(route: Route, test: &'static str, mean: [f64; 3], sd: [f64; 3]) {
    let estimator = match route {
        Route::Dml | Route::DmlComposed => "dml",
        Route::Aipw | Route::AipwComposed => "aipw",
    };
    let n = grid_n(600);
    let mut tally =
        CoverageTally::for_record(RecordKey { test, dgp: DGP, interval: INTERVAL }, REPORTED_LEVEL);
    for row in
        map_replicates(n_sim(), |rep| replicate_on(route, n, grid_seed(93_000 + rep), mean, sd))
    {
        let Some((interval, rows)) = row else {
            tally.skip();
            continue;
        };
        tally.bind_validation(
            &construction(estimator),
            ScopeFacts {
                row_count: rows,
                replicates_ok: None,
                posterior_draws: None,
                unidentified_mass: 0.0,
            },
            "not_applicable",
        );
        tally.record(Some(interval), TRUTH);
    }
    tally.assert();
}

/// Each scored interval is finite and ordered, and the fit is centred near the effect.
#[test]
fn the_scored_interval_is_ordered_and_centred_near_the_effect() {
    let ((lo, hi), rows) = replicate(1_500, 93_001, [0.0, 0.0, 0.0], [10.0, 10.0, 10.0])
        .expect("a prior-regularized fit");
    assert_eq!(rows, 1_500);
    assert!(lo < hi);
    // One replicate: the centre lies within four standard errors of the effect (whether a
    // single interval covers is what the ignored coverage cells measure, not a smoke test).
    let se = (hi - lo) / (2.0 * 1.959_963_984_540_054);
    let centre = 0.5 * (lo + hi);
    assert!((centre - TRUTH).abs() < 4.0 * se, "[{lo}, {hi}] against the effect {TRUTH}");
}

/// A weak prior (zero mean, standard deviation 10): the near-unpenalized outcome fit.
#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn dml_outcome_prior_weak_interval() {
    measure(Route::Dml, "dml_outcome_prior_weak_interval", [0.0, 0.0, 0.0], [10.0, 10.0, 10.0]);
}

/// An informative prior centred on the true outcome slopes (`z0` 1, `z1` 0.5) with the
/// intercept, which differs by arm, left wide.
#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn dml_outcome_prior_informative_interval() {
    measure(
        Route::Dml,
        "dml_outcome_prior_informative_interval",
        [0.0, 1.0, 0.5],
        [10.0, 0.1, 0.1],
    );
}

/// A tight prior on badly wrong coefficients: the outcome nuisance is far from the truth and
/// the correctly specified propensity carries the estimate.
#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn dml_outcome_prior_misspecified_interval() {
    measure(
        Route::Dml,
        "dml_outcome_prior_misspecified_interval",
        [3.0, -2.0, 4.0],
        [0.3, 0.3, 0.3],
    );
}

/// The AIPW estimator route under a weak prior (zero mean, standard deviation 10).
#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn aipw_outcome_prior_weak_interval() {
    measure(Route::Aipw, "aipw_outcome_prior_weak_interval", [0.0, 0.0, 0.0], [10.0, 10.0, 10.0]);
}

/// The AIPW estimator route under a prior centred on the true outcome slopes.
#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn aipw_outcome_prior_informative_interval() {
    measure(
        Route::Aipw,
        "aipw_outcome_prior_informative_interval",
        [0.0, 1.0, 0.5],
        [10.0, 0.1, 0.1],
    );
}

/// The AIPW estimator route under a tight prior on badly wrong coefficients.
#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn aipw_outcome_prior_misspecified_interval() {
    measure(
        Route::Aipw,
        "aipw_outcome_prior_misspecified_interval",
        [3.0, -2.0, 4.0],
        [0.3, 0.3, 0.3],
    );
}

/// The DML route under a composed prior with named coordinates, centred on the true slopes.
#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn dml_composed_prior_informative_interval() {
    measure(
        Route::DmlComposed,
        "dml_composed_prior_informative_interval",
        [0.0, 1.0, 0.5],
        [10.0, 0.1, 0.1],
    );
}

/// The DML route under a composed prior on badly wrong coefficients.
#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn dml_composed_prior_misspecified_interval() {
    measure(
        Route::DmlComposed,
        "dml_composed_prior_misspecified_interval",
        [3.0, -2.0, 4.0],
        [0.3, 0.3, 0.3],
    );
}

/// The AIPW estimator route under a composed prior centred on the true slopes.
#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn aipw_composed_prior_informative_interval() {
    measure(
        Route::AipwComposed,
        "aipw_composed_prior_informative_interval",
        [0.0, 1.0, 0.5],
        [10.0, 0.1, 0.1],
    );
}
