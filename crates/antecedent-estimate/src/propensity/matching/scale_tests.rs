//! Propensity matching at tens of thousands of donors against a quadratic reference.
//!
//! The matching index answers every query exactly (sorted distinct values in one dimension),
//! so the matched donors and the bias-corrected difference of every query must equal those of
//! an exhaustive scan that takes every donor at the nearest distance and averages them. Each test fits the estimator on a seeded population at its stated size, checks a
//! sample of queries against the scan, checks a whole estimate on a subsample against the
//! scan-based estimate, and replays the fit.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::time::{Duration, Instant};

use antecedent_core::{AssumptionSet, ExecutionContext};

use super::*;
use crate::propensity::scale_fixture::{Design, Population, att_query};

const RUNTIME_LIMIT: Duration = Duration::from_secs(120);
const MEMORY_LIMIT_BYTES: u64 = 4 * 1024 * 1024 * 1024;
/// Rows of the subsample whose whole estimate is recomputed by the scan.
const SUBSAMPLE_ROWS: usize = 5_000;

/// The two arms of a prepared problem on the logit propensity score.
struct Arms {
    treated_feat: Vec<f64>,
    treated_y: Vec<f64>,
    control_feat: Vec<f64>,
    control_y: Vec<f64>,
}

fn estimator(caliper: Option<f64>) -> PropensityMatching {
    let est = PropensityMatching::new().with_bootstrap_replicates(0);
    match caliper {
        Some(limit) => est.with_caliper(limit),
        None => est,
    }
}

fn arms(est: &PropensityMatching, prep: &PreparedPropensityProblem) -> Arms {
    let mut workspace = PropensityEstimationWorkspace::default();
    let model =
        PropensityModel::fit(prep, est.backend, &mut workspace.propensity, &est.glm_options)
            .expect("the propensity model fits");
    let score = model.logit_scores(prep);
    let (treated_idx, control_idx) = split_by_treatment(&prep.treatment);
    Arms {
        treated_feat: gather(&score, &treated_idx),
        treated_y: gather(&prep.outcome, &treated_idx),
        control_feat: gather(&score, &control_idx),
        control_y: gather(&prep.outcome, &control_idx),
    }
}

/// The scan: every donor at the nearest distance, ascending by position, and that distance.
fn scan_nearest(donors: &[f64], query: f64) -> (Vec<usize>, f64) {
    let mut best = f64::INFINITY;
    let mut set = Vec::new();
    for (position, &donor) in donors.iter().enumerate() {
        let distance = (query - donor).abs();
        if distance < best {
            best = distance;
            set = vec![position];
        } else if distance.to_bits() == best.to_bits() {
            set.push(position);
        }
    }
    (set, best)
}

/// The scan's donor set and bias-corrected difference for treated query `q`, `None` outside the
/// caliper.
fn scan_match(
    arms: &Arms,
    beta: Option<&[f64]>,
    q: usize,
    caliper: Option<f64>,
) -> Option<(Vec<usize>, f64, f64)> {
    let query = arms.treated_feat[q];
    let (set, distance) = scan_nearest(&arms.control_feat, query);
    if set.is_empty() || !distance.is_finite() || caliper.is_some_and(|limit| distance > limit) {
        return None;
    }
    let mean = |value: &dyn Fn(usize) -> f64| set.iter().map(|&d| value(d)).sum::<f64>() / set.len() as f64;
    let raw = arms.treated_y[q] - mean(&|d| arms.control_y[d]);
    let bias = match beta {
        Some(beta) => {
            predict_linear(beta, &[query], 1, 0)
                - mean(&|d| predict_linear(beta, &arms.control_feat, 1, d))
        }
        None => 0.0,
    };
    Some((set.clone(), raw - bias, distance))
}

/// Two sums of the same terms in another order agree to rounding.
fn same_to_rounding(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-12 * (1.0 + a.abs().max(b.abs()))
}

/// What a sample of queries showed against the scan.
struct Agreement {
    checked: usize,
    matched_by_scan: usize,
    /// Most donors the scan found at the nearest distance for one sampled query.
    most_tied: usize,
    /// Sampled queries whose nearest donor sits at distance zero.
    zero_distance: usize,
    /// Queries matched over the whole arm.
    matched_in_total: usize,
}

/// Match every treated query through the index and compare every `stride`-th with the scan.
fn assert_queries_equal_the_scan(
    arms: &Arms,
    caliper: Option<f64>,
    stride: usize,
    ctx: &ExecutionContext,
) -> Agreement {
    let mut workspace = PropensityEstimationWorkspace::default();
    let matched = match_diffs(
        MatchSide { features: &arms.control_feat, outcome: &arms.control_y },
        MatchSide { features: &arms.treated_feat, outcome: &arms.treated_y },
        1,
        MatchingDistance::Absolute,
        1,
        caliper,
        &mut workspace,
        ctx,
    )
    .expect("the sorted index matches");
    let (donor_sets, diffs, queries) = (matched.donor_sets(), matched.diffs.clone(), matched.queries.clone());
    let beta = fit_linear_mean(&arms.control_feat, &arms.control_y, 1);
    let mut agreement = Agreement {
        checked: 0,
        matched_by_scan: 0,
        most_tied: 0,
        zero_distance: 0,
        matched_in_total: queries.len(),
    };
    for q in (0..arms.treated_y.len()).step_by(stride) {
        let expected = scan_match(arms, beta.as_deref(), q, caliper);
        let got = queries.binary_search(&q).ok().map(|slot| {
            let mut set = donor_sets[slot].to_vec();
            set.sort_unstable();
            (set, diffs[slot])
        });
        match (expected, got) {
            (None, None) => {}
            (Some((set, diff, distance)), Some((got_set, got_diff))) => {
                assert_eq!(got_set, set, "query {q}: the index chose other donors");
                assert!(same_to_rounding(got_diff, diff), "query {q}: {got_diff} vs {diff}");
                agreement.matched_by_scan += 1;
                agreement.most_tied = agreement.most_tied.max(set.len());
                agreement.zero_distance += usize::from(distance.to_bits() == 0);
            }
            (expected, got) => {
                panic!("query {q}: scan {expected:?}, index {got:?} disagree on matching");
            }
        }
        agreement.checked += 1;
    }
    agreement
}

/// The scan-based ATT of `population`'s own fit: the mean over matched treated queries of the
/// bias-corrected difference to the scan's donor.
fn assert_estimate_equals_the_scan(
    population: &Population,
    caliper: Option<f64>,
    ctx: &ExecutionContext,
) {
    let (data, estimand) = population.table();
    let est = estimator(caliper);
    let prep = est.prepare(&data, &estimand, &att_query()).expect("prepared");
    let effect = est
        .fit(&prep, &mut PropensityEstimationWorkspace::default(), ctx, AssumptionSet::new())
        .expect("the subsample fits");
    let arms = arms(&est, &prep);
    let beta = fit_linear_mean(&arms.control_feat, &arms.control_y, 1);
    let diffs: Vec<f64> = (0..arms.treated_y.len())
        .filter_map(|q| scan_match(&arms, beta.as_deref(), q, caliper).map(|(_, diff, _)| diff))
        .collect();
    assert!(!diffs.is_empty());
    let reference = diffs.iter().sum::<f64>() / diffs.len() as f64;
    assert!(same_to_rounding(effect.ate, reference), "ATT {} differs from the scan's {reference}", effect.ate);
    assert_eq!(effect.provenance.n_obs, Some(diffs.len() as u64));
}

/// How the caliper of a run is chosen.
#[derive(Clone, Copy)]
enum Caliper {
    None,
    /// The median nearest-donor distance of the sampled queries, so about half are refused.
    MedianNearest,
}

struct Run {
    agreement: Agreement,
    treated: usize,
    controls: usize,
    /// Distinct propensity scores among the controls.
    distinct_control_scores: usize,
}

/// The shared scale workload: fit at the design's size, replay, compare with the scan.
fn exercise(design: Design, caliper: Caliper, stride: usize) -> Run {
    let population = design.population();
    let (data, estimand) = population.table();
    let ctx = ExecutionContext::for_tests(design.seed);
    let unconstrained = estimator(None);
    let prep = unconstrained.prepare(&data, &estimand, &att_query()).expect("prepared");
    let full_arms = arms(&unconstrained, &prep);
    assert_eq!(full_arms.treated_y.len(), design.treated);
    assert_eq!(full_arms.control_y.len(), design.rows - design.treated);
    let caliper = match caliper {
        Caliper::None => None,
        Caliper::MedianNearest => {
            let mut nearest: Vec<f64> = (0..full_arms.treated_y.len())
                .step_by(stride)
                .map(|q| scan_nearest(&full_arms.control_feat, full_arms.treated_feat[q]).1)
                .collect();
            nearest.sort_by(f64::total_cmp);
            Some(nearest[nearest.len() / 2])
        }
    };

    let est = estimator(caliper);
    let started = Instant::now();
    let mut workspace = PropensityEstimationWorkspace::default();
    let first = est.fit(&prep, &mut workspace, &ctx, AssumptionSet::new()).expect("fits");
    let elapsed = started.elapsed();
    assert!(elapsed < RUNTIME_LIMIT, "{} donors took {elapsed:?}", design.rows);
    let retained = first.provenance.retained_memory_bytes.expect("retained memory is reported");
    assert!(retained < MEMORY_LIMIT_BYTES, "retained {retained} bytes");

    // Deterministic replay: a fresh workspace, the workspace holding the index, and a
    // multi-threaded context give the identical estimate and standard error.
    let fresh = est
        .fit(&prep, &mut PropensityEstimationWorkspace::default(), &ctx, AssumptionSet::new())
        .expect("replays");
    let reused = est.fit(&prep, &mut workspace, &ctx, AssumptionSet::new()).expect("replays");
    let threaded = est
        .fit(
            &prep,
            &mut PropensityEstimationWorkspace::default(),
            &ExecutionContext::production(design.seed, 4),
            AssumptionSet::new(),
        )
        .expect("replays");
    assert_eq!(workspace.matching_index_builds, 1, "the second fit rebuilt the index");
    for replay in [&fresh, &reused, &threaded] {
        assert_eq!(replay.ate.to_bits(), first.ate.to_bits());
        assert_eq!(replay.se_analytic().to_bits(), first.se_analytic().to_bits());
        assert_eq!(replay.provenance.n_obs, first.provenance.n_obs);
    }

    let agreement = assert_queries_equal_the_scan(&full_arms, caliper, stride, &ctx);
    assert_eq!(first.provenance.n_obs, Some(agreement.matched_in_total as u64));
    assert_estimate_equals_the_scan(&population.first(SUBSAMPLE_ROWS), caliper, &ctx);
    let mut scores: Vec<u64> = full_arms.control_feat.iter().map(|s| s.to_bits()).collect();
    scores.sort_unstable();
    scores.dedup();
    Run {
        agreement,
        treated: design.treated,
        controls: full_arms.control_y.len(),
        distinct_control_scores: scores.len(),
    }
}

fn design(rows: usize, treated: usize, covariates: usize, seed: u64) -> Design {
    Design { rows, treated, covariates, seed, lattice: None, duplicated: false }
}

#[test]
fn sorted_exact_one_dimensional_propensity_matching_20000_donors() {
    let run = exercise(design(20_000, 2_000, 3, 1_598_979), Caliper::None, 10);
    assert_eq!(run.agreement.checked, 200);
    assert_eq!(run.agreement.matched_by_scan, run.agreement.checked);
    assert_eq!(run.agreement.matched_in_total, run.treated);
}

#[test]
fn sorted_exact_one_dimensional_propensity_matching_100000_donors() {
    let run = exercise(design(100_000, 10_000, 3, 7_674_962), Caliper::None, 50);
    assert_eq!(run.agreement.checked, 200);
    assert_eq!(run.agreement.matched_by_scan, run.agreement.checked);
    assert_eq!(run.agreement.matched_in_total, run.treated);
}

#[test]
fn sorted_matching_averages_over_ties() {
    // One covariate on a quarter-step lattice: the fitted score takes a few dozen values, so
    // thousands of donors share the nearest distance and the match is all of them.
    let design = Design { lattice: Some(0.25), ..design(20_000, 2_000, 1, 7_297_538) };
    let run = exercise(design, Caliper::None, 7);
    assert_eq!(run.agreement.matched_by_scan, run.agreement.checked);
    assert!(run.distinct_control_scores < 100, "{} distinct scores", run.distinct_control_scores);
    assert!(run.agreement.most_tied >= 100, "most tied donors {}", run.agreement.most_tied);
    assert!(run.agreement.zero_distance > 0);
}

#[test]
fn sorted_matching_preserves_calipers() {
    let run = exercise(design(20_000, 2_000, 3, 3_706_826), Caliper::MedianNearest, 10);
    assert!(run.agreement.matched_in_total > 0, "the caliper refused every unit");
    assert!(
        run.agreement.matched_in_total < run.treated,
        "the caliper refused no unit: {} of {}",
        run.agreement.matched_in_total,
        run.treated
    );
    assert!(run.agreement.matched_by_scan < run.agreement.checked);
}

#[test]
fn sorted_matching_preserves_duplicates() {
    // Every row of the second half repeats the covariates of an earlier row, so donors
    // coincide exactly and the sorted search must still return every duplicate.
    let design = Design { duplicated: true, ..design(20_000, 2_000, 2, 8_730_913) };
    let run = exercise(design, Caliper::None, 5);
    assert_eq!(run.agreement.matched_by_scan, run.agreement.checked);
    assert!(
        run.distinct_control_scores * 10 < run.controls * 7,
        "{} distinct scores among {} controls",
        run.distinct_control_scores,
        run.controls
    );
    assert!(run.agreement.zero_distance > 0, "no sampled query met a duplicate donor");
}

#[test]
fn sorted_matching_preserves_deterministic_replay() {
    let design = design(20_000, 2_000, 3, 9_941_869);
    // The generator and the whole fit replay bit for bit from the seed.
    let fit = || {
        let (data, estimand) = design.population().table();
        let est = estimator(None);
        let prep = est.prepare(&data, &estimand, &att_query()).expect("prepared");
        est.fit(
            &prep,
            &mut PropensityEstimationWorkspace::default(),
            &ExecutionContext::for_tests(design.seed),
            AssumptionSet::new(),
        )
        .expect("fits")
    };
    let (first, second) = (fit(), fit());
    assert_eq!(first.ate.to_bits(), second.ate.to_bits());
    assert_eq!(first.se_analytic().to_bits(), second.se_analytic().to_bits());
    assert_eq!(first.provenance.n_obs, second.provenance.n_obs);
    // And the replay equals the reference path (also exercises the threaded and reused fits).
    let run = exercise(design, Caliper::None, 20);
    assert_eq!(run.agreement.matched_by_scan, run.agreement.checked);
}
