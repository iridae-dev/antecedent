//! Coverage of the 95% normal interval `effect ± z · se_analytic` of the full-population
//! standardized contrast of a three-level observational treatment (the cross-fitted
//! multinomial generalized-propensity AIPW with its analytic influence-function standard
//! error).
//!
//! Three treatment levels are drawn from a multinomial logit of one covariate, and the outcome
//! is `effects[level] + 0.6 z + 0.5 noise` with `effects = [0, 0.4, 0.9]`, so the effect of
//! level 2 against level 0 is 0.9 over every population. The scored interval is the route's
//! own candidate interval (`multi_level_normal_interval` of the facade report); the facade
//! publishes it only once a coverage record of this construction exists, so the design scores
//! the candidate and the plain tests pin that the route keys the same construction.
//!
//! `multilevel_generalized_propensity_interval` is registered in `scripts/gate_calibration.sh`
//! (`run_ml`) and its coverage is unmeasured: the coverage record is the only missing piece.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::float_cmp,
    clippy::needless_range_loop,
    reason = "the interval is compared with its own arithmetic; the fixture indexes columns"
)]

mod common;

use antecedent::analysis::multi_level_treatment::{
    AdjustmentColumn, MULTI_LEVEL_INTERVAL_WITHHELD, MULTI_LEVEL_TREATMENT_CALIBRATION,
    MultiLevelReportWire, MultiLevelSpecWire, MultiLevelTargetWire, MultiLevelTreatment,
    MultiLevelTreatmentRequest, multi_level_interval_basis, multi_level_normal_interval,
};
use antecedent::result::{REPORTED_SE_INTERVAL_LEVEL, reported_se_interval_z};
use antecedent_core::{ExecutionContext, StreamDomain};
use antecedent_kernels::standard_normal;
use common::calibration::{
    Construction, CoverageTally, REPORTED_LEVEL, RecordKey, ScopeFacts, grid_n, map_replicates,
    n_sim,
};

const INTERVAL: &str = "analytic_se";
const LEVELS: [f64; 3] = [0.0, 1.0, 2.0];
const LEVEL_EFFECTS: [f64; 3] = [0.0, 0.4, 0.9];
/// Multinomial-logit intercept and slope of levels 1 and 2 against level 0.
const LOGIT: [(f64, f64); 2] = [(0.3, 0.6), (-0.2, -0.5)];
const OUTCOME_SLOPE: f64 = 0.6;
const TRUTH: f64 = 0.9;
const FOLDS: usize = 5;
const BASE_SEED: u64 = 9_224_800;

/// One simulated dataset of `n` rows: the treatment level, the outcome and the covariate.
fn simulate(n: usize, seed: u64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut rng = ExecutionContext::for_tests(seed).rng.stream_for(StreamDomain::Test, 0x31);
    let (mut t, mut y, mut z) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        let zi = standard_normal(&mut rng);
        let weights =
            [1.0, (LOGIT[0].0 + LOGIT[0].1 * zi).exp(), (LOGIT[1].0 + LOGIT[1].1 * zi).exp()];
        let u = rng.next_f64() * weights.iter().sum::<f64>();
        let level = if u < weights[0] {
            0
        } else if u < weights[0] + weights[1] {
            1
        } else {
            2
        };
        z[i] = zi;
        t[i] = LEVELS[level];
        y[i] = LEVEL_EFFECTS[level] + OUTCOME_SLOPE * zi + 0.5 * standard_normal(&mut rng);
    }
    (t, y, z)
}

/// Fit the full-population contrast of level 2 against level 0 through the facade.
fn fit(n: usize, seed: u64) -> Option<MultiLevelReportWire> {
    let (treatment, outcome, z) = simulate(n, seed);
    MultiLevelTreatment::evaluate(&MultiLevelTreatmentRequest {
        outcome,
        treatment,
        adjustment: vec![AdjustmentColumn { name: "z".into(), values: z }],
        spec: MultiLevelSpecWire {
            target: MultiLevelTargetWire::FullPopulation,
            active: 2.0,
            control: 0.0,
            levels: Some(LEVELS.to_vec()),
            folds: None,
            seed,
        },
    })
    .ok()
}

/// One replicate: the candidate interval of the facade report with its row count.
fn replicate(n: usize, seed: u64) -> Option<((f64, f64), u64)> {
    let report = fit(n, seed)?;
    Some((multi_level_normal_interval(report.effect, report.se_analytic)?, report.n_obs))
}

fn construction() -> Construction {
    Construction {
        query: "MultiLevelContrast".into(),
        graph_class: "NotApplicable".into(),
        structure: "fixed".into(),
        modality: "tabular".into(),
        inference: "Frequentist".into(),
        estimator: "propensity.generalized_aipw".into(),
        interval_method: INTERVAL.into(),
        se_kind: "influence".into(),
        dependence: "iid".into(),
        posterior: String::new(),
        functional: format!(
            "multi_level.full_population.contrast;levels={};folds={FOLDS}",
            LEVELS.len()
        ),
        identification: "point".into(),
        reported_level: REPORTED_LEVEL,
    }
}

/// The design binds exactly the construction the route keys, at the route's reported level.
#[test]
fn the_design_construction_is_the_one_the_route_keys() {
    let basis = multi_level_interval_basis(MultiLevelTargetWire::FullPopulation, 3, FOLDS, 3000);
    let design = construction();
    let key = &basis.key;
    assert_eq!(key.query, design.query);
    assert_eq!(key.graph_class, design.graph_class);
    assert_eq!(key.structure, design.structure);
    assert_eq!(key.modality, design.modality);
    assert_eq!(key.inference, design.inference);
    assert_eq!(key.estimator, design.estimator);
    assert_eq!(key.interval_method, design.interval_method);
    assert_eq!(key.se_kind, design.se_kind);
    assert_eq!(key.dependence, design.dependence);
    assert_eq!(key.posterior, design.posterior);
    assert_eq!(key.functional, design.functional);
    assert_eq!(key.identification, design.identification);
    assert_eq!(key.level, REPORTED_SE_INTERVAL_LEVEL);
    assert_eq!(design.reported_level, REPORTED_SE_INTERVAL_LEVEL);
    assert_eq!(basis.scope.row_count, 3000);
}

/// A published interval is `effect ± z · se` of the report; without a covering record the
/// interval is withheld as `scope_not_assessed` and the standard error stays unmeasured.
#[test]
fn a_published_interval_is_the_estimate_plus_or_minus_z_standard_errors_or_withheld() {
    let report = fit(3000, BASE_SEED).expect("a full-population fit");
    assert_eq!(report.n_obs, 3000);
    let half = reported_se_interval_z() * report.se_analytic;
    let candidate = multi_level_normal_interval(report.effect, report.se_analytic).unwrap();
    assert_eq!(candidate, (report.effect - half, report.effect + half));
    assert!(
        candidate.0 < TRUTH + 4.0 * report.se_analytic
            && candidate.1 > TRUTH - 4.0 * report.se_analytic
    );
    if let Some(interval) = &report.interval {
        assert_eq!((interval.lower, interval.upper), candidate);
        assert_eq!(interval.level, REPORTED_SE_INTERVAL_LEVEL);
        assert!(!interval.coverage_record.is_empty());
        assert_eq!(report.interval_status, "calibrated");
        assert_eq!(report.calibration, "calibrated");
        assert!(report.interval_reason.is_none());
    } else {
        assert_eq!(report.interval_status, MULTI_LEVEL_INTERVAL_WITHHELD);
        assert_eq!(report.calibration, MULTI_LEVEL_TREATMENT_CALIBRATION);
        assert!(report.interval_reason.is_some());
    }
}

/// The pairwise subpopulation target has no measurement design, so its interval stays withheld.
#[test]
fn the_pairwise_target_publishes_no_interval() {
    let (treatment, outcome, z) = simulate(1500, BASE_SEED);
    let report = MultiLevelTreatment::evaluate(&MultiLevelTreatmentRequest {
        outcome,
        treatment,
        adjustment: vec![AdjustmentColumn { name: "z".into(), values: z }],
        spec: MultiLevelSpecWire {
            target: MultiLevelTargetWire::PairwiseSubpopulation,
            active: 2.0,
            control: 0.0,
            levels: None,
            folds: None,
            seed: 1,
        },
    })
    .expect("a pairwise fit");
    assert!(report.interval.is_none());
    assert_eq!(report.interval_status, MULTI_LEVEL_INTERVAL_WITHHELD);
    assert_eq!(report.calibration, MULTI_LEVEL_TREATMENT_CALIBRATION);
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn multilevel_generalized_propensity_interval() {
    let n = grid_n(3000);
    let mut tally = CoverageTally::for_record(
        RecordKey {
            test: "multilevel_generalized_propensity_interval",
            dgp: "confounded_three_level_treatment",
            interval: INTERVAL,
        },
        REPORTED_LEVEL,
    );
    for row in map_replicates(n_sim(), |rep| replicate(n, BASE_SEED + rep)) {
        let Some((interval, rows)) = row else {
            tally.skip();
            continue;
        };
        tally.bind(
            &construction(),
            ScopeFacts {
                row_count: rows,
                replicates_ok: None,
                posterior_draws: None,
                unidentified_mass: 0.0,
            },
        );
        tally.record(Some(interval), TRUTH);
    }
    tally.assert();
}
