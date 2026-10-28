//! Covariate-distance matching at ten and a hundred thousand donors with eight covariates.
//!
//! The k-d tree answers every query exactly, so the match set must equal the exhaustive
//! scan's, and the estimate must be the mean of the differences to those matches. The
//! workload runs under a hard memory limit, retains memory linear in the donor rows and
//! refuses (rather than allocating) when the limit is below what the index needs.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::time::{Duration, Instant};

use antecedent_core::{AssumptionSet, ExecutionContext};
use antecedent_stats::{StatsError, nearest_euclidean_scalar};

use super::*;
use crate::propensity::matching::{MatchSide, MatchedDiffs, match_diffs};
use crate::propensity::prepare::{gather_rowmajor, split_by_treatment};
use crate::propensity::scale_fixture::{Design, att_query};

const COVARIATES: usize = 8;
const RUNTIME_LIMIT: Duration = Duration::from_secs(900);
const MEMORY_LIMIT_BYTES: u64 = 8 * 1024 * 1024 * 1024;
/// Retained bytes allowed per donor feature value: the index keeps the features, the tree
/// and per-query buffers, a small constant multiple of the raw features and nothing
/// quadratic in the donors.
const RETAINED_BYTES_PER_FEATURE_VALUE: u64 = 64 * 8;

fn bounded(seed: u64, hard_limit_bytes: u64) -> ExecutionContext {
    let mut ctx = ExecutionContext::for_tests(seed);
    ctx.set_memory_limit(Some(hard_limit_bytes));
    ctx
}

fn exercise(rows: usize, seed: u64, stride: usize) {
    let population = Design {
        rows,
        treated: rows / 10,
        covariates: COVARIATES,
        seed,
        lattice: None,
        duplicated: false,
    }
    .population();
    let (data, estimand) = population.table();
    let est = DistanceMatching::new().with_bootstrap_replicates(0);
    let prep = est.prepare(&data, &estimand, &att_query()).expect("prepared");
    let ctx = bounded(seed, MEMORY_LIMIT_BYTES);

    let started = Instant::now();
    let mut workspace = PropensityEstimationWorkspace::default();
    let effect = est.fit(&prep, &mut workspace, &ctx, AssumptionSet::new()).expect("fits");
    let elapsed = started.elapsed();
    assert!(elapsed < RUNTIME_LIMIT, "{rows} donors took {elapsed:?}");
    let retained = effect.provenance.retained_memory_bytes.expect("retained memory is reported");
    let raw_values = (rows * COVARIATES) as u64;
    assert!(
        retained < MEMORY_LIMIT_BYTES && retained < RETAINED_BYTES_PER_FEATURE_VALUE * raw_values,
        "retained {retained} bytes for {raw_values} feature values"
    );

    // The match set: the sample covariates on the standardized scale the estimator uses.
    let mut features = to_row_major(&prep.covariates, prep.nrows);
    standardize_rowmajor_inplace(&mut features, prep.nrows, COVARIATES);
    let (treated_idx, control_idx) = split_by_treatment(&prep.treatment);
    let treated_feat = gather_rowmajor(&features, COVARIATES, &treated_idx);
    let control_feat = gather_rowmajor(&features, COVARIATES, &control_idx);
    let treated_y: Vec<f64> = treated_idx.iter().map(|&i| prep.outcome[i]).collect();
    let control_y: Vec<f64> = control_idx.iter().map(|&i| prep.outcome[i]).collect();
    let mut match_workspace = PropensityEstimationWorkspace::default();
    let matched = match_diffs(
        MatchSide { features: &control_feat, outcome: &control_y },
        MatchSide { features: &treated_feat, outcome: &treated_y },
        COVARIATES,
        MatchingDistance::Euclidean,
        1,
        None,
        &mut match_workspace,
        &ctx,
    )
    .expect("the bounded match runs");
    let donors: Vec<usize> = matched.donor_sets().iter().map(|set| set[0]).collect();
    let MatchedDiffs { diffs, queries, .. } = matched;
    assert_eq!(queries.len(), treated_idx.len(), "every treated unit finds a donor");
    for q in (0..treated_idx.len()).step_by(stride) {
        let query = &treated_feat[q * COVARIATES..(q + 1) * COVARIATES];
        let (donor, _) =
            nearest_euclidean_scalar(query, &control_feat, control_idx.len(), COVARIATES)
                .expect("donors exist");
        assert_eq!(donors[q], donor, "treated unit {q}: the tree chose another donor");
    }
    // The estimate is the mean of the differences to that match set.
    let mean = diffs.iter().sum::<f64>() / diffs.len() as f64;
    assert_eq!(effect.ate.to_bits(), mean.to_bits());
    assert_eq!(effect.provenance.n_obs, Some(diffs.len() as u64));

    // Replay: a second fit on the retained index is bit-identical.
    let again = est.fit(&prep, &mut workspace, &ctx, AssumptionSet::new()).expect("replays");
    assert_eq!(workspace.matching_index_builds, 1);
    assert_eq!(again.ate.to_bits(), effect.ate.to_bits());
    assert_eq!(again.se_analytic().to_bits(), effect.se_analytic().to_bits());

    // Below what the index needs, the fit refuses instead of allocating.
    let tight = bounded(seed, 64 * 1024);
    let refused = est
        .fit(&prep, &mut PropensityEstimationWorkspace::default(), &tight, AssumptionSet::new())
        .expect_err("a 64 KiB budget cannot hold the index");
    assert!(
        matches!(refused, EstimationError::Stats(StatsError::ResourceLimit { .. })),
        "{refused:?}"
    );
}

#[test]
fn multivariate_distance_matching_bounded_resources_10000() {
    exercise(10_000, 4_237_894, 10);
}

#[test]
fn multivariate_distance_matching_bounded_resources_100000() {
    exercise(100_000, 3_506_513, 50);
}
