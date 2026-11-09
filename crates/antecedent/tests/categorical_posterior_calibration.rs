//! Prior-predictive coverage of the bounded conjugate categorical contrast.
//!
//! This ignored calibration harness measures nothing until explicitly run. Its truth is
//! drawn from the declared Normal-Inverse-Gamma prior, then a homoskedastic Gaussian
//! sample is drawn conditionally. It makes no causal or misspecification guarantee.
//!
//! One record is measured for every level count from two to eight. Each record's replicates
//! cycle through nine settings, three row-count multipliers crossed with three arm
//! imbalances, on top of the run's sample-size grid point, so a record covers the row range
//! the three grid points span and the arm shapes in it. A record is keyed by its level count
//! because the runtime never extrapolates across level counts.

#![cfg(feature = "calibration-internal")]

#[path = "common/calibration.rs"]
mod calibration;
#[path = "../../antecedent-estimate/tests/common/candidate_calibration.rs"]
mod candidate;

use antecedent::analysis::categorical_posterior_candidate::{
    CategoricalPosteriorCandidateRequest, CategoricalSamplingPremise, prepare,
    replay_calibration_candidate,
};
use antecedent_core::categorical_candidate_contract::{
    CREDIBLE_LEVEL, LEVEL_PRECISION, MIN_LEVEL_ROWS, MIN_LEVELS, NUISANCE_PRECISION,
    VARIANCE_SCALE, VARIANCE_SHAPE,
};
use antecedent_core::{CalibrationBasis, ExecutionContext, sampling_design::SamplingDesign};
use antecedent_io::categorical_treatment_artifact::{
    CategoricalPosteriorDeclaration, CategoricalSpecWire, CategoricalTreatmentRequest,
    CovarianceKindWire, LevelScaleWire,
};
use calibration::{
    CoverageTally, RecordKey, gaussian, grid_n, grid_seed, map_replicates, n_sim, stream_seed,
    uniform,
};

/// Base row count; the run's grid point halves or doubles it.
const BASE_ROWS: usize = 128;
/// Row multipliers cycled across replicates. Together with the grid point they span 64 to
/// 4096 rows, the row counts a record covers.
const ROW_FACTORS: [usize; 3] = [1, 4, 16];
const IMBALANCES: usize = 3;
/// Most level counts measured; a record covers only the level counts it measured.
const MAX_LEVELS: usize = 8;

/// Arm sizes: every level keeps the candidate's minimum and the remaining rows follow the
/// imbalance profile, which is balanced, halves per level, or leaves the minimum to every
/// non-reference level.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "each share is a non-negative fraction of the spare rows, far below usize::MAX"
)]
fn arm_sizes(rows: usize, levels: usize, imbalance: usize) -> Vec<usize> {
    let weights: Vec<f64> = (0..levels)
        .map(|level| match imbalance {
            0 => 1.0,
            1 => 0.5_f64.powi(i32::try_from(level).expect("at most eight levels")),
            _ => f64::from(u8::from(level == 0)),
        })
        .collect();
    let spare = rows - MIN_LEVEL_ROWS * levels;
    let total: f64 = weights.iter().sum();
    let mut sizes: Vec<usize> =
        weights.iter().map(|w| MIN_LEVEL_ROWS + (spare as f64 * w / total) as usize).collect();
    let assigned: usize = sizes.iter().sum();
    sizes[0] += rows - assigned;
    sizes
}

type Replicate = Option<(String, [f64; 2], CalibrationBasis, f64)>;

/// One prior-predictive dataset with `levels` levels; scores one non-reference contrast.
fn replicate(rep: u64, levels: usize) -> Replicate {
    let seed = stream_seed(grid_seed(0x6361_7400 + rep), levels as u64);
    let setting = usize::try_from(rep).ok()?;
    let rows = grid_n(BASE_ROWS) * ROW_FACTORS[setting % ROW_FACTORS.len()];
    let imbalance = (setting / ROW_FACTORS.len()) % IMBALANCES;
    let target = 1 + (setting / (ROW_FACTORS.len() * IMBALANCES)) % (levels - 1);
    let mut uniform = uniform(seed, 19);
    let mut normal = gaussian(seed ^ 0x7e11);
    // The shape is two, so G, a sum of two Exp(1), is Gamma(2, 1) and scale / G is InvGamma(2, scale).
    assert!((VARIANCE_SHAPE - 2.0).abs() < f64::EPSILON);
    let variance = VARIANCE_SCALE / (-(1.0 - uniform()).ln() - (1.0 - uniform()).ln());
    let scale = variance.sqrt();
    let intercept = scale * normal() / NUISANCE_PRECISION.sqrt();
    let effects: Vec<f64> = (0..levels)
        .map(|level| if level == 0 { 0.0 } else { scale * normal() / LEVEL_PRECISION.sqrt() })
        .collect();
    let declared: Vec<String> = (0..levels).map(|level| format!("level-{level}")).collect();
    let sizes = arm_sizes(rows, levels, imbalance);
    let mut row_levels = Vec::with_capacity(rows);
    let mut outcome = Vec::with_capacity(rows);
    for (level, &size) in sizes.iter().enumerate() {
        for _ in 0..size {
            row_levels.push(declared[level].clone());
            outcome.push(intercept + effects[level] + scale * normal());
        }
    }
    let request = CategoricalPosteriorCandidateRequest {
        data: CategoricalTreatmentRequest {
            outcome,
            row_snapshot: "prior-predictive".into(),
            adjustment: vec![],
            levels: row_levels,
            spec: CategoricalSpecWire {
                reference: declared[0].clone(),
                declared_levels: declared.clone(),
                scale: LevelScaleWire::Unordered,
                min_level_rows: 2,
                pairwise: vec![],
                monotonicity: None,
                covariance: CovarianceKindWire::ModelBased,
            },
        },
        posterior: CategoricalPosteriorDeclaration {
            level_precision: LEVEL_PRECISION,
            nuisance_precision: NUISANCE_PRECISION,
            variance_shape: VARIANCE_SHAPE,
            variance_scale: VARIANCE_SCALE,
            credible_level: CREDIBLE_LEVEL,
        },
        sampling: CategoricalSamplingPremise {
            design: SamplingDesign::IndependentSubjectTrajectories,
            subject_ids: (0..rows).map(|i| format!("subject-{i}")).collect(),
        },
    };
    let ctx = ExecutionContext::for_tests(seed);
    let prepared = prepare(request.clone(), &ctx).ok()?;
    let (intervals, basis, receipt) = prepared.calibration_candidate(&ctx).ok()?;
    drop(prepared);
    let (replayed, rebound) = replay_calibration_candidate(request, receipt, &ctx).ok()?;
    assert_eq!(intervals, replayed);
    assert_eq!(basis, rebound);
    assert_eq!(basis.row_count, rows as u64);
    let (_, interval) = intervals.iter().find(|(name, _)| *name == declared[target])?;
    Some((declared[target].clone(), *interval, basis, effects[target]))
}

#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn categorical_conjugate_contrast_prior_predictive_interval() {
    let runs: Vec<Vec<Replicate>> = map_replicates(n_sim(), |rep| {
        (MIN_LEVELS..=MAX_LEVELS).map(|levels| replicate(rep, levels)).collect()
    });
    for (offset, levels) in (MIN_LEVELS..=MAX_LEVELS).enumerate() {
        let mut tally = CoverageTally::for_record(
            RecordKey {
                test: "categorical_conjugate_contrast_prior_predictive_interval",
                dgp: "declared_levels_normal_inverse_gamma_prior_predictive_imbalance_grid",
                interval: "posterior_quantile",
            },
            0.95,
        )
        .labelled(format!("k{levels}"));
        for run in &runs {
            let Some((_, interval, basis, truth)) = &run[offset] else {
                tally.skip();
                continue;
            };
            candidate::bind(&mut tally, basis);
            tally.record(Some((interval[0], interval[1])), *truth);
        }
        assert_eq!(
            tally.record_id().as_deref(),
            Some(
                format!(
                    "cov.categorical_treatment_contrast.not_applicable.bayesian.posterior_quantile.l95.categorical_conjugate_contrast_prior_predictive_interval.v2_validation_not_applicable.k{levels}"
                )
                .as_str()
            )
        );
        tally.assert();
    }
}
