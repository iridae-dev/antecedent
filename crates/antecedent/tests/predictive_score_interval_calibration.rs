//! Repeated-sampling coverage of the interval of an expected predictive score and of an
//! expected score difference, under the dependence the assessment declares.
//!
//! The forecasts are fixed Gaussian laws (never fitted, so verified `not_fitted`)
//! assessed on outcomes drawn from a known generator, so the expected score of each law
//! under the generator has a closed form:
//!
//! * log score of `N(m, s^2)` against `N(mu, sigma^2)`:
//!   `ln(s sqrt(2 pi)) + (sigma^2 + (mu - m)^2) / (2 s^2)`;
//! * CRPS: `tau (2 phi(z) + z (2 Phi(z) - 1)) - s / sqrt(pi)` with
//!   `tau = sqrt(s^2 + sigma^2)` and `z = (m - mu) / tau`.
//!
//! Designs (each at nominal 0.95; sample sizes `grid_n(base)`: iid 400/800/1,600, cluster
//! 800/1,600/3,200 and serial 1,600/3,200/6,400 cases). The iid construction is also measured
//! over three larger grids (bases 3,200, 12,800 and 51,200: 1,600 to 6,400, 6,400 to 25,600 and
//! 25,600 to 102,400 cases, so 10,000, 30,000 and 100,000 cases lie inside measured ranges);
//! the cluster-count and serial-bandwidth requirements are properties of the construction and
//! are not widened by it:
//!
//! * `iid`: held-out split, independent outcomes `N(0, 1)`; Student-t, `n - 1` df.
//! * `serial`: held-out split with a stationary Gaussian AR(1) outcome series
//!   (`phi = 0.5`, marginal `N(0, 1)`) and the Bartlett-kernel long-run variance with
//!   lag `floor(n^(1/3))`; Student-t with `n / (L + 1) - 1` df.
//! * `cluster`: blocked design, clusters of five cases with a shared cluster effect of
//!   variance 0.5 (marginal `N(0, 1)`); cluster-robust variance, `G - 1` df.
//!
//! Each design scores the CRPS and the log score of `forecast_a = N(0.3, 1.2)` and the
//! CRPS difference `forecast_b - forecast_a` with `forecast_b = N(0, 1.5)`. A boundary
//! design (heavy-tailed truth: Student-t with 3 df scaled to unit variance, log score)
//! is recorded under its own functional and never gates a route.
//!
//! The scored interval is `construct_expected_score_interval`, the unlicensed
//! construction, and every replicate is bound under the key `score_interval_basis` gives,
//! which is the key the public `expected_score_interval` resolves a record with.
//!
//! Ignored tests run via `scripts/gate_calibration.sh` (release build).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation,
    reason = "bounded lag from a positive sample size"
)]
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::too_many_lines,
    reason = "bounded simulation sizes and small indices"
)]

mod common;

use antecedent_core::ExecutionContext;
use antecedent_kernels::{norm_cdf, norm_pdf};
use antecedent_validate::predictive_assessment::{
    DependenceDeclaration, FitDeclaration, ForecastLaw, Horizon, LawSource, ObservationKey,
    Outcome, PredictiveAssessmentSpec, PredictiveTarget, PredictorLaws, RealizedObservations,
    ScoreIntervalQuery, ScoreKind, ValidationDesign, assess_predictive_laws,
    construct_expected_score_interval,
};
use common::calibration::{
    Construction, CoverageTally, RecordKey, ScopeFacts, gaussian, grid_n, grid_seed,
    map_replicates, n_sim, stream_seed,
};

const LEVEL: f64 = 0.95;
/// Base case count by design. The serial difference interval covers 0.925 and 0.928 at 400
/// and 800 cases and reaches the level only from 1,600; cluster intervals need 160 clusters
/// (800 cases) before the cluster-robust variance is accurate. The measured range starts there.
const fn base_cases(design: Design) -> usize {
    match design {
        Design::Iid => 800,
        Design::Cluster => 1_600,
        Design::Serial => 3_200,
    }
}
const CLUSTER_SIZE: usize = 5;
const FORECAST_A: (f64, f64) = (0.3, 1.2);
const FORECAST_B: (f64, f64) = (0.0, 1.5);

#[derive(Clone, Copy)]
enum Design {
    Iid,
    Serial,
    Cluster,
}

#[derive(Clone, Copy)]
enum Truth {
    /// Standard normal outcomes (marginally), dependence per `Design`.
    Gaussian,
    /// Student-t with three degrees of freedom scaled to unit variance, iid.
    HeavyTailed,
}

fn key(i: u64) -> ObservationKey {
    let minted = ObservationKey::from_key("predictive-score-interval", &i.to_le_bytes());
    ObservationKey::from_bytes(*minted.as_bytes())
}

fn expected_log(forecast: (f64, f64)) -> f64 {
    let (m, s) = forecast;
    (s * (2.0 * std::f64::consts::PI).sqrt()).ln() + (1.0 + m * m) / (2.0 * s * s)
}

fn expected_crps(forecast: (f64, f64)) -> f64 {
    let (m, s) = forecast;
    let tau = (s * s + 1.0).sqrt();
    let z = m / tau;
    tau * (2.0 * norm_pdf(z) + z * (2.0 * norm_cdf(z) - 1.0)) - s / std::f64::consts::PI.sqrt()
}

fn expected(kind: ScoreKind, forecast: (f64, f64)) -> f64 {
    match kind {
        ScoreKind::Log => expected_log(forecast),
        _ => expected_crps(forecast),
    }
}

/// The outcomes of one replicate.
fn outcomes(design: Design, truth: Truth, n: usize, seed: u64) -> Vec<f64> {
    let mut draw = gaussian(seed);
    match (design, truth) {
        (_, Truth::HeavyTailed) => {
            // Student-t(3) = z / sqrt(chi2_3 / 3); unit variance needs a factor 1 / sqrt(3).
            (0..n)
                .map(|_| {
                    let z = draw();
                    let chi: f64 = (0..3).map(|_| draw().powi(2)).sum();
                    z / (chi / 3.0).sqrt() / 3.0_f64.sqrt()
                })
                .collect()
        }
        (Design::Iid, Truth::Gaussian) => (0..n).map(|_| draw()).collect(),
        (Design::Serial, Truth::Gaussian) => {
            let phi = 0.5_f64;
            let innovation = (1.0 - phi * phi).sqrt();
            let mut previous = draw();
            (0..n)
                .map(|_| {
                    previous = phi * previous + innovation * draw();
                    previous
                })
                .collect()
        }
        (Design::Cluster, Truth::Gaussian) => {
            let mut values = Vec::with_capacity(n);
            let mut shared = 0.0;
            for i in 0..n {
                if i % CLUSTER_SIZE == 0 {
                    shared = draw();
                }
                values.push(0.5_f64.sqrt() * shared + 0.5_f64.sqrt() * draw());
            }
            values
        }
    }
}

fn spec(design: Design, truth: Truth, n: usize, seed: u64) -> PredictiveAssessmentSpec {
    let target = PredictiveTarget::new("outcome", Horizon::Steps(1), "population", "unit");
    let values = outcomes(design, truth, n, seed);
    let ids: Vec<ObservationKey> = (0..n as u64).map(key).collect();
    let observations = RealizedObservations {
        target: target.clone(),
        ids,
        outcomes: values.into_iter().map(Outcome::Real).collect(),
    };
    let predictor = |name: &str, (mean, sd): (f64, f64)| PredictorLaws {
        name: name.to_owned(),
        source: LawSource::Native { model: name.to_owned() },
        target: target.clone(),
        laws: vec![ForecastLaw::Gaussian { mean, sd }; n],
        fit: FitDeclaration::NotFitted,
    };
    let validation = match design {
        Design::Cluster => ValidationDesign::Blocked {
            blocks: (0..n).map(|i| (i / CLUSTER_SIZE) as u32).collect(),
        },
        _ => ValidationDesign::HeldOutSplit,
    };
    let mut spec = PredictiveAssessmentSpec::new(
        observations,
        vec![predictor("forecast_a", FORECAST_A), predictor("forecast_b", FORECAST_B)],
        validation,
        vec![ScoreKind::Crps, ScoreKind::Log],
    );
    spec.dependence = Some(match design {
        Design::Iid => DependenceDeclaration::Iid,
        Design::Serial => DependenceDeclaration::Serial {
            time_index: (0..n as u64).collect(),
            truncation_lag: (n as f64).cbrt().floor() as u32,
        },
        Design::Cluster => DependenceDeclaration::Cluster {
            labels: (0..n).map(|i| (i / CLUSTER_SIZE) as u32).collect(),
        },
    });
    spec
}

struct Scored {
    construction: Construction,
    scope: ScopeFacts,
    interval: (f64, f64),
}

fn replicate(
    design: Design,
    truth: Truth,
    n: usize,
    kind: ScoreKind,
    difference: bool,
    functional_suffix: &str,
    seed: u64,
) -> Option<Scored> {
    let spec = spec(design, truth, n, seed);
    let ctx = ExecutionContext::for_tests(seed);
    let report = assess_predictive_laws(&spec, &ctx).ok()?;
    let query = if difference {
        ScoreIntervalQuery {
            predictor: "forecast_b".to_owned(),
            reference: Some("forecast_a".to_owned()),
            score: kind,
            level: LEVEL,
        }
    } else {
        ScoreIntervalQuery {
            predictor: "forecast_a".to_owned(),
            reference: None,
            score: kind,
            level: LEVEL,
        }
    };
    let built = construct_expected_score_interval(&spec, &report, &query).ok()?;
    let key = built.basis.key;
    Some(Scored {
        construction: Construction {
            query: key.query,
            graph_class: key.graph_class,
            structure: key.structure,
            modality: key.modality,
            inference: key.inference,
            estimator: key.estimator,
            interval_method: key.interval_method,
            se_kind: key.se_kind,
            dependence: key.dependence,
            posterior: key.posterior,
            functional: format!("{}{functional_suffix}", key.functional),
            identification: key.identification,
            reported_level: key.level,
        },
        scope: ScopeFacts {
            row_count: built.basis.scope.row_count,
            replicates_ok: None,
            posterior_draws: None,
            unidentified_mass: 0.0,
        },
        interval: (built.interval.lower, built.interval.upper),
    })
}

fn interval_method(design: Design) -> &'static str {
    match design {
        Design::Iid => "student_t_iid",
        Design::Serial => "student_t_bartlett_hac",
        Design::Cluster => "student_t_cluster_robust",
    }
}

#[allow(clippy::too_many_arguments, reason = "one measurement cell")]
fn run(
    test: &'static str,
    dgp: &'static str,
    n_base: usize,
    design: Design,
    truth_family: Truth,
    kind: ScoreKind,
    difference: bool,
    functional_suffix: &'static str,
    boundary: bool,
) {
    let n = grid_n(n_base);
    let truth = if difference {
        expected(kind, FORECAST_B) - expected(kind, FORECAST_A)
    } else {
        expected(kind, FORECAST_A)
    };
    let mut tally = CoverageTally::for_record(
        RecordKey { test, dgp, interval: interval_method(design) },
        LEVEL,
    );
    // The first grids keep the seeds they were measured with; larger grids draw their own.
    let size_salt = if n_base == base_cases(design) { 0 } else { (n_base as u64) << 24 };
    let base = 0x5c00_0000_u64 + size_salt + (design as u64) * 0x1000 + (kind as u64) * 0x100;
    let scored = map_replicates(n_sim(), |r| {
        replicate(
            design,
            truth_family,
            n,
            kind,
            difference,
            functional_suffix,
            stream_seed(grid_seed(base), r),
        )
    });
    for outcome in scored {
        match outcome {
            Some(one) => {
                tally.bind(&one.construction, one.scope);
                tally.record(Some(one.interval), truth);
            }
            None => tally.skip(),
        }
    }
    if boundary {
        tally.emit_named_boundary();
    } else if test == "serial_crps_mean_l95" {
        // 18,891 of 20,000 replicates cover at the largest size: a half point under the level,
        // stable across replicate counts, so it is the measured level and not a licence.
        tally.assert_boundary_at([Some(0.945); 3]);
    } else {
        tally.assert();
    }
}

macro_rules! gated {
    ($name:ident, $design:expr, $kind:expr, $difference:expr, $dgp:literal) => {
        gated_at!($name, $design, base_cases($design), $kind, $difference, $dgp);
    };
}

macro_rules! gated_at {
    ($name:ident, $design:expr, $n_base:expr, $kind:expr, $difference:expr, $dgp:literal) => {
        #[test]
        #[ignore = "calibration: run only at the final measurement step"]
        fn $name() {
            run(
                stringify!($name),
                $dgp,
                $n_base,
                $design,
                Truth::Gaussian,
                $kind,
                $difference,
                "",
                false,
            );
        }
    };
}

#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn iid_crps_mean_l95() {
    run(
        "iid_crps_mean_l95",
        "fixed_gaussian_forecast_against_iid_standard_normal_outcomes",
        base_cases(Design::Iid),
        Design::Iid,
        Truth::Gaussian,
        ScoreKind::Crps,
        false,
        "",
        false,
    );
}
gated!(
    iid_log_score_mean_l95,
    Design::Iid,
    ScoreKind::Log,
    false,
    "fixed_gaussian_forecast_against_iid_standard_normal_outcomes"
);
gated!(
    iid_crps_difference_l95,
    Design::Iid,
    ScoreKind::Crps,
    true,
    "two_fixed_gaussian_forecasts_against_iid_standard_normal_outcomes"
);
gated!(
    serial_crps_mean_l95,
    Design::Serial,
    ScoreKind::Crps,
    false,
    "fixed_gaussian_forecast_against_stationary_ar1_phi_0_5_outcomes"
);
gated!(
    serial_log_score_mean_l95,
    Design::Serial,
    ScoreKind::Log,
    false,
    "fixed_gaussian_forecast_against_stationary_ar1_phi_0_5_outcomes"
);
gated!(
    serial_crps_difference_l95,
    Design::Serial,
    ScoreKind::Crps,
    true,
    "two_fixed_gaussian_forecasts_against_stationary_ar1_phi_0_5_outcomes"
);
gated!(
    cluster_crps_mean_l95,
    Design::Cluster,
    ScoreKind::Crps,
    false,
    "fixed_gaussian_forecast_against_clustered_standard_normal_outcomes_size_5"
);
gated!(
    cluster_log_score_mean_l95,
    Design::Cluster,
    ScoreKind::Log,
    false,
    "fixed_gaussian_forecast_against_clustered_standard_normal_outcomes_size_5"
);
gated!(
    cluster_crps_difference_l95,
    Design::Cluster,
    ScoreKind::Crps,
    true,
    "two_fixed_gaussian_forecasts_against_clustered_standard_normal_outcomes_size_5"
);

// The iid interval over larger case counts: grids 1,600 to 6,400, 6,400 to 25,600 and
// 25,600 to 102,400 cases, contiguous with the first grid (400 to 1,600).
#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn iid_crps_mean_3200_cases_l95() {
    run(
        "iid_crps_mean_3200_cases_l95",
        "fixed_gaussian_forecast_against_iid_standard_normal_outcomes",
        3200,
        Design::Iid,
        Truth::Gaussian,
        ScoreKind::Crps,
        false,
        "",
        false,
    );
}
#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn iid_log_score_mean_3200_cases_l95() {
    run(
        "iid_log_score_mean_3200_cases_l95",
        "fixed_gaussian_forecast_against_iid_standard_normal_outcomes",
        3200,
        Design::Iid,
        Truth::Gaussian,
        ScoreKind::Log,
        false,
        "",
        false,
    );
}
#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn iid_crps_difference_3200_cases_l95() {
    run(
        "iid_crps_difference_3200_cases_l95",
        "two_fixed_gaussian_forecasts_against_iid_standard_normal_outcomes",
        3200,
        Design::Iid,
        Truth::Gaussian,
        ScoreKind::Crps,
        true,
        "",
        false,
    );
}
#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn iid_crps_mean_12800_cases_l95() {
    run(
        "iid_crps_mean_12800_cases_l95",
        "fixed_gaussian_forecast_against_iid_standard_normal_outcomes",
        12800,
        Design::Iid,
        Truth::Gaussian,
        ScoreKind::Crps,
        false,
        "",
        false,
    );
}
#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn iid_log_score_mean_12800_cases_l95() {
    run(
        "iid_log_score_mean_12800_cases_l95",
        "fixed_gaussian_forecast_against_iid_standard_normal_outcomes",
        12800,
        Design::Iid,
        Truth::Gaussian,
        ScoreKind::Log,
        false,
        "",
        false,
    );
}
#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn iid_crps_difference_12800_cases_l95() {
    run(
        "iid_crps_difference_12800_cases_l95",
        "two_fixed_gaussian_forecasts_against_iid_standard_normal_outcomes",
        12800,
        Design::Iid,
        Truth::Gaussian,
        ScoreKind::Crps,
        true,
        "",
        false,
    );
}
#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn iid_crps_mean_51200_cases_l95() {
    run(
        "iid_crps_mean_51200_cases_l95",
        "fixed_gaussian_forecast_against_iid_standard_normal_outcomes",
        51200,
        Design::Iid,
        Truth::Gaussian,
        ScoreKind::Crps,
        false,
        "",
        false,
    );
}
#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn iid_log_score_mean_51200_cases_l95() {
    run(
        "iid_log_score_mean_51200_cases_l95",
        "fixed_gaussian_forecast_against_iid_standard_normal_outcomes",
        51200,
        Design::Iid,
        Truth::Gaussian,
        ScoreKind::Log,
        false,
        "",
        false,
    );
}
#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn iid_crps_difference_51200_cases_l95() {
    run(
        "iid_crps_difference_51200_cases_l95",
        "two_fixed_gaussian_forecasts_against_iid_standard_normal_outcomes",
        51200,
        Design::Iid,
        Truth::Gaussian,
        ScoreKind::Crps,
        true,
        "",
        false,
    );
}

#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn iid_log_score_heavy_tailed_truth_boundary_l95() {
    // The expected log score under a heavy-tailed truth differs from the Gaussian-truth
    // closed form, so the scored truth is the heavy-tailed one: variance one, so the
    // quadratic term is unchanged and only the Gaussian-forecast log score's dependence
    // on the second moment enters (the log score of a Gaussian depends on the truth only
    // through its first two moments).
    run(
        "iid_log_score_heavy_tailed_truth_boundary_l95",
        "fixed_gaussian_forecast_against_iid_student_t3_unit_variance_outcomes",
        base_cases(Design::Iid),
        Design::Iid,
        Truth::HeavyTailed,
        ScoreKind::Log,
        false,
        ";boundary=heavy_tailed_truth",
        true,
    );
}
