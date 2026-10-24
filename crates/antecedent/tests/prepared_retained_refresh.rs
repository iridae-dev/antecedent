//! Retained summary refresh of a prepared study, through the public facade.
//!
//! The estimator a prepared study froze decides which summary family its result rests on
//! (integer-count least squares, real-weight least squares, scatter, conjugate blocks). Each
//! change (append, window, evict, weight, mask, replace) must leave the summary equal to a
//! fresh reconstruction and to a reference computed here from the model rows by the closed
//! forms, never by calling the library's solver. Reduced retention refuses what it cannot
//! serve with the next attempt, and a disk-backed result is admitted against the operation's
//! pool, evicted under the registry's budget and reloaded to the same answer.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

mod common;

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

use antecedent::bootstrap_retention::RetainedBootstrap;
use antecedent::retained_refresh::{
    RefreshFamily, RefreshMutation, RefreshObservation, RefreshResult, RetainedRefresh,
    spill_registry_within_admission,
};
use antecedent::strategy_table::EstimatorId;
use antecedent::{AcceptedGraph, CausalError, InferenceMode, PreparedStudy, RefuteSuite, Study};
use antecedent_core::execution::AdmissionPool;
use antecedent_core::{CausalQuery, ExecutionContext};
use antecedent_state::retention::{RetainedOperation, RetentionMode};
use antecedent_state::{ConjugateOlsPrior, RefreshOutcome};
use common::persistent_dgp::{autoregressive_pulse_dag, autoregressive_pulse_series, pulse};

const TOLERANCE: f64 = 1e-9;

fn observation(id: u64, weight: f64) -> RefreshObservation {
    let x = 0.25 * id as f64 - 1.5;
    let y = 0.5 + 1.75 * x + ((id * 7) % 5) as f64 * 0.1 - 0.2;
    RefreshObservation::weighted(id, vec![1.0, x], y, weight)
}

fn observations(ids: std::ops::Range<u64>, fractional: bool) -> Vec<RefreshObservation> {
    ids.map(|id| {
        let base = 1.0 + (id % 3) as f64;
        observation(id, if fractional { base * 0.5 } else { base })
    })
    .collect()
}

fn live(row: &RefreshObservation) -> bool {
    !row.masked && row.weight > 0.0
}

/// Intercept and slope of the weighted simple regression, by the closed form.
fn regression(rows: &[RefreshObservation]) -> (f64, f64) {
    let rows: Vec<&RefreshObservation> = rows.iter().filter(|row| live(row)).collect();
    let weight: f64 = rows.iter().map(|row| row.weight).sum();
    let mean_x = rows.iter().map(|row| row.weight * row.x[1]).sum::<f64>() / weight;
    let mean_y = rows.iter().map(|row| row.weight * row.y).sum::<f64>() / weight;
    let sxx: f64 = rows.iter().map(|row| row.weight * (row.x[1] - mean_x).powi(2)).sum();
    let sxy: f64 = rows.iter().map(|row| row.weight * (row.x[1] - mean_x) * (row.y - mean_y)).sum();
    let slope = sxy / sxx;
    (mean_y - slope * mean_x, slope)
}

fn apply_to_model(model: &mut Vec<RefreshObservation>, mutation: &RefreshMutation) {
    match mutation {
        RefreshMutation::Append { rows, .. } => model.extend(rows.iter().cloned()),
        RefreshMutation::Window { max_rows } => {
            let drop_count = model.len().saturating_sub(*max_rows);
            model.drain(..drop_count);
        }
        RefreshMutation::Evict { id } => model.retain(|row| row.id != *id),
        RefreshMutation::SetWeight { id, weight } => {
            model.iter_mut().filter(|row| row.id == *id).for_each(|row| row.weight = *weight);
        }
        RefreshMutation::SetMask { id, masked } => {
            model.iter_mut().filter(|row| row.id == *id).for_each(|row| row.masked = *masked);
        }
        RefreshMutation::Replace { rows, .. } => *model = rows.clone(),
        RefreshMutation::Block { .. } => unreachable!("not a block family"),
    }
}

fn coefficients(result: &RefreshResult) -> Vec<f64> {
    match result {
        RefreshResult::Ols(snapshot) => snapshot.beta.clone().unwrap(),
        RefreshResult::Weighted(snapshot) => snapshot.beta.clone().unwrap(),
        other => panic!("not a least-squares result: {other:?}"),
    }
}

fn assert_close(left: f64, right: f64, what: &str) {
    assert!((left - right).abs() <= TOLERANCE * (1.0 + right.abs()), "{what}: {left} vs {right}");
}

fn source(name: &str) -> Arc<str> {
    Arc::from(name)
}

fn least_squares_script(fractional: bool) -> Vec<(RefreshMutation, RefreshOutcome)> {
    let weight = if fractional { 4.5 } else { 4.0 };
    vec![
        (
            RefreshMutation::Append {
                source: source("prepared"),
                rows: observations(10..16, fractional),
            },
            RefreshOutcome::Updated,
        ),
        (RefreshMutation::SetWeight { id: 3, weight }, RefreshOutcome::Recomputed),
        (RefreshMutation::Evict { id: 1 }, RefreshOutcome::Recomputed),
        (RefreshMutation::SetMask { id: 5, masked: true }, RefreshOutcome::Recomputed),
        (RefreshMutation::Window { max_rows: 9 }, RefreshOutcome::Recomputed),
        (
            RefreshMutation::Replace {
                source: source("other"),
                rows: observations(20..28, fractional),
            },
            RefreshOutcome::Recomputed,
        ),
    ]
}

fn run_least_squares(family: RefreshFamily, fractional: bool) {
    let initial = observations(0..10, fractional);
    let mut refresh = RetainedRefresh::start(family, "prepared", 2, &initial, None).unwrap();
    assert_eq!(refresh.family(), family);
    assert_eq!(refresh.retention(), RetentionMode::FullReusable);
    let mut model = initial;
    for (mutation, expected) in least_squares_script(fractional) {
        let before = refresh.recomputes();
        let outcome = refresh.apply_verified(&mutation).unwrap();
        assert_eq!(outcome, expected, "{mutation:?}");
        let rebuilt = u64::from(expected == RefreshOutcome::Recomputed);
        assert_eq!(refresh.recomputes(), before + rebuilt, "{mutation:?}");
        apply_to_model(&mut model, &mutation);
        let (intercept, slope) = regression(&model);
        let beta = coefficients(&refresh.result().unwrap());
        assert_close(beta[0], intercept, "intercept");
        assert_close(beta[1], slope, "slope");
        let fresh = refresh.fresh_result().unwrap();
        assert_eq!(fresh.source_digest(), refresh.result().unwrap().source_digest());
        let fresh_beta = coefficients(&fresh);
        assert_close(fresh_beta[1], slope, "fresh slope");
    }
}

#[test]
fn integer_count_least_squares_follows_every_change_and_equals_the_closed_form() {
    run_least_squares(RefreshFamily::Ols, false);
}

#[test]
fn real_weight_least_squares_follows_every_change_and_equals_the_closed_form() {
    run_least_squares(RefreshFamily::WeightedOls, true);
}

#[test]
fn a_change_the_family_cannot_carry_or_a_malformed_row_leaves_the_summary_unchanged() {
    let initial = observations(0..10, false);
    let mut refresh =
        RetainedRefresh::start(RefreshFamily::Ols, "prepared", 2, &initial, None).unwrap();
    let before = refresh.result().unwrap();
    let fractional =
        refresh.apply_verified(&RefreshMutation::SetWeight { id: 3, weight: 1.5 }).unwrap_err();
    assert!(fractional.to_string().contains("integer count"), "{fractional}");
    let repeated = refresh
        .apply_verified(&RefreshMutation::Append {
            source: source("prepared"),
            rows: vec![observation(4, 1.0)],
        })
        .unwrap_err();
    assert!(repeated.to_string().contains("already in the summary"), "{repeated}");
    let other_source = refresh
        .apply_verified(&RefreshMutation::Append {
            source: source("elsewhere"),
            rows: vec![observation(40, 1.0)],
        })
        .unwrap_err();
    assert!(matches!(other_source, CausalError::State(_)), "{other_source}");
    let unknown = refresh.apply_verified(&RefreshMutation::Evict { id: 999 }).unwrap_err();
    assert!(matches!(unknown, CausalError::State(_)), "{unknown}");
    let blocked = refresh
        .apply_verified(&RefreshMutation::in_block("a", RefreshMutation::Evict { id: 1 }))
        .unwrap_err();
    assert!(blocked.to_string().contains("conjugate"), "{blocked}");
    assert_eq!(refresh.result().unwrap(), before);

    // A real weight is carried by the weighted family; the same change succeeds there.
    let mut weighted =
        RetainedRefresh::start(RefreshFamily::WeightedOls, "prepared", 2, &initial, None).unwrap();
    weighted.apply_verified(&RefreshMutation::SetWeight { id: 3, weight: 1.5 }).unwrap();
}

fn pairs(ids: std::ops::Range<u64>) -> Vec<RefreshObservation> {
    ids.map(|id| {
        let a = (id as f64 * 0.7).sin() * 2.0;
        let b = 0.5 * a + ((id * 5) % 7) as f64 * 0.2;
        RefreshObservation::new(id, vec![a, b], 0.0)
    })
    .collect()
}

fn sample_covariance(rows: &[RefreshObservation]) -> f64 {
    let live: Vec<&RefreshObservation> = rows.iter().filter(|row| !row.masked).collect();
    let n = live.len() as f64;
    let mean_a = live.iter().map(|row| row.x[0]).sum::<f64>() / n;
    let mean_b = live.iter().map(|row| row.x[1]).sum::<f64>() / n;
    live.iter().map(|row| (row.x[0] - mean_a) * (row.x[1] - mean_b)).sum::<f64>() / (n - 1.0)
}

#[test]
fn scatter_follows_every_change_and_equals_the_closed_form() {
    let initial = pairs(0..12);
    let mut refresh =
        RetainedRefresh::start(RefreshFamily::Covariance, "prepared", 2, &initial, None).unwrap();
    let mut model = initial;
    let script = [
        (
            RefreshMutation::Append { source: source("prepared"), rows: pairs(12..20) },
            RefreshOutcome::Updated,
        ),
        (RefreshMutation::Evict { id: 2 }, RefreshOutcome::Recomputed),
        (RefreshMutation::SetMask { id: 6, masked: true }, RefreshOutcome::Recomputed),
        (RefreshMutation::Window { max_rows: 14 }, RefreshOutcome::Recomputed),
        (
            RefreshMutation::Replace { source: source("other"), rows: pairs(30..41) },
            RefreshOutcome::Recomputed,
        ),
    ];
    for (mutation, expected) in script {
        assert_eq!(refresh.apply_verified(&mutation).unwrap(), expected, "{mutation:?}");
        apply_to_model(&mut model, &mutation);
        let RefreshResult::Covariance(snapshot) = refresh.result().unwrap() else {
            panic!("not a scatter result");
        };
        let covariance = snapshot.sample_covariance.unwrap();
        assert_close(covariance[1], sample_covariance(&model), "covariance");
        assert_eq!(snapshot.n, model.iter().filter(|row| !row.masked).count() as u64);
    }
    // The scatter carries no weights: the refusal names the next attempt and changes nothing.
    let before = refresh.result().unwrap();
    let weight = refresh.apply_verified(&RefreshMutation::SetWeight { id: 30, weight: 2.0 });
    assert!(weight.unwrap_err().to_string().contains("no weights"));
    let weighted_row = RefreshObservation::weighted(99, vec![0.0, 0.0], 0.0, 2.0);
    let append = refresh.apply_verified(&RefreshMutation::Append {
        source: source("other"),
        rows: vec![weighted_row],
    });
    assert!(append.unwrap_err().to_string().contains("no weights"));
    assert_eq!(refresh.result().unwrap(), before);
}

fn prior() -> ConjugateOlsPrior {
    ConjugateOlsPrior::new(vec![0.0, 0.0], vec![0.5, 0.0, 0.0, 0.5], 3.0, 2.0).unwrap()
}

/// Posterior mean `(P0 + X'WX)^-1 (P0 m0 + X'Wy)` over every block, with `P0 = 0.5 I`, `m0 = 0`.
fn posterior_mean(blocks: &BTreeMap<String, Vec<RefreshObservation>>) -> [f64; 2] {
    let (mut a, mut b) = ([[0.5, 0.0], [0.0, 0.5]], [0.0, 0.0]);
    for row in blocks.values().flatten().filter(|row| live(row)) {
        for i in 0..2 {
            b[i] += row.weight * row.x[i] * row.y;
            for j in 0..2 {
                a[i][j] += row.weight * row.x[i] * row.x[j];
            }
        }
    }
    let det = a[0][0] * a[1][1] - a[0][1] * a[1][0];
    [(a[1][1] * b[0] - a[0][1] * b[1]) / det, (a[0][0] * b[1] - a[1][0] * b[0]) / det]
}

fn block_mutation(
    blocks: &mut BTreeMap<String, Vec<RefreshObservation>>,
    label: &str,
    mutation: &RefreshMutation,
) {
    if let RefreshMutation::Replace { rows, .. } = mutation {
        blocks.clear();
        blocks.insert(label.to_owned(), rows.clone());
    } else {
        apply_to_model(blocks.entry(label.to_owned()).or_default(), mutation);
    }
}

#[test]
fn conjugate_blocks_follow_every_change_and_equal_the_closed_form_posterior() {
    let first = observations(0..8, false);
    let prior = prior();
    let mut refresh = RetainedRefresh::start(
        RefreshFamily::ConjugateBlocks,
        "prepared",
        2,
        &first,
        Some((&prior, "a")),
    )
    .unwrap();
    let mut blocks = BTreeMap::new();
    blocks.insert("a".to_owned(), first);
    let script = [
        ("b", RefreshMutation::Append { source: source("later"), rows: observations(0..6, false) }),
        (
            "a",
            RefreshMutation::Append {
                source: source("prepared"),
                rows: observations(8..12, false),
            },
        ),
        ("a", RefreshMutation::Evict { id: 2 }),
        ("b", RefreshMutation::SetWeight { id: 1, weight: 5.0 }),
        ("a", RefreshMutation::SetMask { id: 4, masked: true }),
        ("b", RefreshMutation::Window { max_rows: 4 }),
        (
            "c",
            RefreshMutation::Replace { source: source("fresh"), rows: observations(0..9, false) },
        ),
    ];
    let expected = [
        RefreshOutcome::Updated,
        RefreshOutcome::Recomputed,
        RefreshOutcome::Recomputed,
        RefreshOutcome::Recomputed,
        RefreshOutcome::Recomputed,
        RefreshOutcome::Recomputed,
        RefreshOutcome::Recomputed,
    ];
    for ((label, mutation), expected) in script.into_iter().zip(expected) {
        let wrapped = RefreshMutation::in_block(label, mutation.clone());
        assert_eq!(refresh.apply_verified(&wrapped).unwrap(), expected, "{wrapped:?}");
        block_mutation(&mut blocks, label, &mutation);
        let RefreshResult::Conjugate { summary, identity } = refresh.result().unwrap() else {
            panic!("not a conjugate result");
        };
        let reference = posterior_mean(&blocks);
        assert_close(summary.mean[0], reference[0], "posterior mean 0");
        assert_close(summary.mean[1], reference[1], "posterior mean 1");
        let RefreshResult::Conjugate { summary: rebuilt, identity: fresh } =
            refresh.fresh_result().unwrap()
        else {
            panic!("not a conjugate result");
        };
        assert_eq!(identity, fresh);
        assert_close(rebuilt.mean[1], reference[1], "fresh posterior mean");
    }
    // The posterior has labelled blocks: an unlabelled change is refused and changes nothing.
    let before = refresh.result().unwrap();
    let unlabelled = refresh.apply_verified(&RefreshMutation::Evict { id: 1 }).unwrap_err();
    assert!(unlabelled.to_string().contains("in_block"), "{unlabelled}");
    assert_eq!(refresh.result().unwrap(), before);
}

fn every_family() -> Vec<RetainedRefresh> {
    let prior = prior();
    vec![
        RetainedRefresh::start(
            RefreshFamily::Ols,
            "prepared",
            2,
            &observations(0..10, false),
            None,
        )
        .unwrap(),
        RetainedRefresh::start(
            RefreshFamily::WeightedOls,
            "prepared",
            2,
            &observations(0..10, true),
            None,
        )
        .unwrap(),
        RetainedRefresh::start(RefreshFamily::Covariance, "prepared", 2, &pairs(0..10), None)
            .unwrap(),
        RetainedRefresh::start(
            RefreshFamily::ConjugateBlocks,
            "prepared",
            2,
            &observations(0..10, false),
            Some((&prior, "a")),
        )
        .unwrap(),
    ]
}

fn evict(family: RefreshFamily, id: u64) -> RefreshMutation {
    let evict = RefreshMutation::Evict { id };
    if family == RefreshFamily::ConjugateBlocks {
        RefreshMutation::in_block("a", evict)
    } else {
        evict
    }
}

fn append(family: RefreshFamily, from: u64) -> RefreshMutation {
    let rows = match family {
        RefreshFamily::Covariance => pairs(from..from + 3),
        _ => observations(from..from + 3, false),
    };
    let append = RefreshMutation::Append { source: source("prepared"), rows };
    if family == RefreshFamily::ConjugateBlocks {
        RefreshMutation::in_block("a", append)
    } else {
        append
    }
}

/// A window that extends the summary without a rebuild: a conjugate posterior takes a new block
/// in place, and rebuilds when rows join an existing block.
fn append_in_place(family: RefreshFamily, from: u64) -> RefreshMutation {
    if family == RefreshFamily::ConjugateBlocks {
        let rows = observations(from..from + 3, false);
        RefreshMutation::in_block(
            "later",
            RefreshMutation::Append { source: source("later"), rows },
        )
    } else {
        append(family, from)
    }
}

#[test]
fn summary_only_retention_extends_in_place_and_refuses_what_needs_rows() {
    for refresh in every_family() {
        let family = refresh.family();
        let mut summary = refresh.clone().into_summary_only();
        assert_eq!(summary.retention(), RetentionMode::SummaryOnly, "{family:?}");
        assert!(summary.require(RetainedOperation::ReuseSummary).is_ok());
        for operation in RetentionMode::SummaryOnly.no_longer_serves() {
            let refusal = summary.require(*operation).unwrap_err();
            assert!(refusal.to_string().contains("summary-only"), "{refusal}");
        }
        // A window of the same source extends the summary without any rows.
        let before = summary.recomputes();
        assert_eq!(summary.apply(&append_in_place(family, 10)).unwrap(), RefreshOutcome::Updated);
        assert_eq!(summary.recomputes(), before);
        // Everything that rebuilds from rows is refused, and the comparison cannot be made.
        let held = summary.result().unwrap();
        let evicted = summary.apply(&evict(family, 1)).unwrap_err();
        assert!(evicted.to_string().contains("summary-only"), "{family:?}: {evicted}");
        assert!(summary.apply_verified(&append(family, 20)).is_err(), "{family:?}");
        assert!(summary.fresh_result().is_err(), "{family:?}");
        assert_eq!(summary.result().unwrap(), held, "{family:?}");
    }
}

struct Scratch(PathBuf);

impl Scratch {
    fn new(name: &str) -> Self {
        let mut dir = std::env::temp_dir();
        dir.push(format!("antecedent-prepared-refresh-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        Self(dir)
    }

    fn at(&self, leaf: &str) -> PathBuf {
        self.0.join(leaf)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn pooled(bytes: u64) -> (ExecutionContext, AdmissionPool) {
    let pool = AdmissionPool::fixed(1, bytes);
    (ExecutionContext::for_tests(5).with_admission(pool.clone()), pool)
}

#[test]
fn a_disk_backed_summary_is_admitted_evicted_reloaded_and_refuses_resident_operations() {
    let scratch = Scratch::new("refresh");
    let (ctx, pool) = pooled(1 << 20);
    let probe = spill_registry_within_admission(&ctx, scratch.at("probe"), 1 << 40).unwrap();
    assert_eq!(probe.budget_bytes(), 1 << 20, "the registry never exceeds the admission budget");
    for refresh in every_family() {
        let family = refresh.family();
        let frame = refresh.spill(&ctx, &probe, "probe").unwrap().handle().bytes();
        assert_eq!(pool.available().bytes(), 1 << 20, "the lease returns after the write");
        // A registry that holds one and a half frames evicts the least recently used.
        let registry =
            spill_registry_within_admission(&ctx, scratch.at(family.as_str()), frame + frame / 2)
                .unwrap();
        let first = refresh.spill(&ctx, &registry, "first").unwrap();
        let second = refresh.spill(&ctx, &registry, "second").unwrap();
        assert_eq!(second.evicted(), ["first"], "{family:?}");
        assert!(!first.on_disk() && second.on_disk(), "{family:?}");
        assert!(first.reload(&ctx).is_err(), "{family:?}: evicted, rebuild from the rows");
        assert_eq!(second.mode(), RetentionMode::DiskBacked);
        let refusal = second.require(RetainedOperation::RefitFromRetainedRows).unwrap_err();
        assert!(refusal.to_string().contains("reload"), "{family:?}: {refusal}");
        second.require(RetainedOperation::ReadDiskArtifact).unwrap();
        // The reload equals the original, and a verified change follows it.
        let mut back = second.reload(&ctx).unwrap();
        assert_eq!(back.result().unwrap(), refresh.result().unwrap(), "{family:?}");
        assert_eq!(back.retention(), RetentionMode::FullReusable);
        back.apply_verified(&append(family, 10)).unwrap();
        back.apply_verified(&evict(family, 1)).unwrap();
        assert_eq!(registry.stats().evictions, 1);
        assert_eq!(registry.stats().reloads, 1);
    }
}

#[test]
fn a_disk_backed_result_is_refused_when_the_operation_cannot_admit_its_frame() {
    let scratch = Scratch::new("refused");
    let (roomy, _) = pooled(1 << 20);
    let registry =
        spill_registry_within_admission(&roomy, scratch.at("registry"), 1 << 20).unwrap();
    let families = every_family();
    let refresh = &families[0];
    let (tight, pool) = pooled(16);
    let refused = refresh.spill(&tight, &registry, "tight").unwrap_err();
    let CausalError::Resource { message } = &refused else { panic!("expected a resource refusal") };
    assert!(message.contains("admission budget"), "{message}");
    assert_eq!(pool.available().bytes(), 16, "nothing was held");
    assert!(!registry.contains("tight"), "nothing was written");
    // Reload is admitted against the same pool: a spill that fit, a reload that does not.
    let spilled = refresh.spill(&roomy, &registry, "kept").unwrap();
    let reload = spilled.reload(&tight).unwrap_err();
    assert!(matches!(reload, CausalError::Resource { .. }), "{reload}");
    assert!(spill_registry_within_admission(&roomy, scratch.at("zero"), 0).is_err());
}

#[test]
fn summary_only_refresh_is_not_spilled_because_it_holds_no_rows() {
    let scratch = Scratch::new("summary");
    let (ctx, _) = pooled(1 << 20);
    let registry = spill_registry_within_admission(&ctx, scratch.at("registry"), 1 << 20).unwrap();
    for refresh in every_family() {
        let summary = refresh.into_summary_only();
        let error = summary.spill(&ctx, &registry, "summary").unwrap_err();
        assert!(error.to_string().contains("summary-only") || error.to_string().contains("rows"));
        assert!(!registry.contains("summary"));
    }
}

fn replicates(count: usize) -> Vec<f64> {
    (0..count).map(|i| 1.0 + ((i * 37) % 11) as f64 * 0.1).collect()
}

fn sample_sd(values: &[f64]) -> f64 {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    (values.iter().map(|value| (value - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt()
}

#[test]
fn bootstrap_results_extend_in_place_and_follow_the_retention_mode() {
    let first = replicates(40);
    let mut result = RetainedBootstrap::from_replicates(first.clone(), 2).unwrap();
    assert_close(result.accounting().se.unwrap(), sample_sd(&first), "bootstrap se");
    assert_eq!(result.accounting().replicates_ok, Some(40));
    assert_eq!(result.accounting().replicates_failed, Some(2));
    let more: Vec<f64> = replicates(70)[40..].to_vec();
    result.append(&more, 1).unwrap();
    let all = replicates(70);
    assert_close(result.accounting().se.unwrap(), sample_sd(&all), "extended se");
    assert_eq!(result.accounting().replicates_failed, Some(3));
    result.check_against_fresh().unwrap();
    assert!(result.append(&[f64::NAN], 0).is_err());
    assert_eq!(result.estimates().len(), 70);

    let ranked: Vec<f64> = (0..=100).map(f64::from).collect();
    let uniform = RetainedBootstrap::from_replicates(ranked, 0).unwrap();
    let (low, high) = uniform.percentile_interval(0.9).unwrap();
    assert_close(low, 5.0, "lower percentile");
    assert_close(high, 95.0, "upper percentile");
    assert!(uniform.percentile_interval(1.5).is_err());

    // Too many failed replicates withhold the standard error, as the estimator's own rule does.
    let failing = RetainedBootstrap::from_replicates(vec![1.0, 2.0], 10).unwrap();
    assert_eq!(failing.accounting().se, None);

    // Summary-only keeps the standard error, extends in place, and refuses the percentile.
    let mut summary = result.clone().into_summary_only();
    assert_eq!(summary.retention(), RetentionMode::SummaryOnly);
    assert!(summary.estimates().is_empty());
    summary.append(&more, 0).unwrap();
    assert!(summary.accounting().se.is_some());
    let refusal = summary.percentile_interval(0.9).unwrap_err();
    assert!(refusal.to_string().contains("summary-only"), "{refusal}");
    assert!(summary.check_against_fresh().is_err());
}

#[test]
fn a_disk_backed_bootstrap_is_admitted_evicted_and_reloaded() {
    let scratch = Scratch::new("bootstrap");
    let (ctx, pool) = pooled(1 << 20);
    let result = RetainedBootstrap::from_replicates(replicates(64), 3).unwrap();
    let probe = spill_registry_within_admission(&ctx, scratch.at("probe"), 1 << 20).unwrap();
    let frame = result.spill(&ctx, &probe, "probe").unwrap().handle().bytes();
    let registry =
        spill_registry_within_admission(&ctx, scratch.at("registry"), frame + frame / 2).unwrap();
    let first = result.spill(&ctx, &registry, "first").unwrap();
    let second = result.spill(&ctx, &registry, "second").unwrap();
    assert_eq!(pool.available().bytes(), 1 << 20);
    assert_eq!(second.evicted(), ["first"]);
    assert!(first.reload(&ctx).is_err());
    let refusal = second.require(RetainedOperation::RowReconstruction).unwrap_err();
    assert!(refusal.to_string().contains("reload"), "{refusal}");
    let back = second.reload(&ctx).unwrap();
    assert_eq!(back.estimates(), result.estimates());
    assert_eq!(back.accounting(), result.accounting());
    back.check_against_fresh().unwrap();

    let (tight, _) = pooled(16);
    assert!(matches!(
        result.spill(&tight, &registry, "tight").unwrap_err(),
        CausalError::Resource { .. }
    ));
    let summary_only = result.clone().into_summary_only();
    assert!(summary_only.spill(&ctx, &registry, "summary").is_err());
}

#[test]
fn the_estimator_decides_which_summary_family_a_study_keeps() {
    use RefreshFamily::{ConjugateBlocks, Covariance, Ols, WeightedOls};
    let cases = [
        (EstimatorId::LinearAdjustmentAte, Some(Ols)),
        (EstimatorId::TemporalLinearAdjustment, Some(Ols)),
        (EstimatorId::PropensityWeighting, Some(WeightedOls)),
        (EstimatorId::Aipw, Some(WeightedOls)),
        (EstimatorId::IvWald, Some(Covariance)),
        (EstimatorId::BayesianGcomp, Some(ConjugateBlocks)),
        (EstimatorId::BayesianTemporalGcomp, Some(ConjugateBlocks)),
        (EstimatorId::PropensityMatching, None),
        (EstimatorId::CausalForest, None),
    ];
    for (id, family) in cases {
        assert_eq!(RefreshFamily::of_estimator(id), family, "{}", id.as_str());
    }
}

fn prepared(inference: InferenceMode) -> PreparedStudy {
    let data = autoregressive_pulse_series(400, 0.6, false, 31);
    let ctx = ExecutionContext::for_tests(11);
    Study::series(data)
        .graph(AcceptedGraph::temporal_dag(autoregressive_pulse_dag(true, false)))
        .query(CausalQuery::TemporalEffect(pulse()))
        .inference(inference)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap()
}

#[test]
fn a_prepared_study_starts_mutates_spills_and_reloads_the_summary_its_estimator_keeps() {
    let study = prepared(InferenceMode::Frequentist);
    let family = study.retained_refresh_family().unwrap();
    assert_eq!(family, RefreshFamily::Ols);
    let initial = observations(0..12, false);
    let mut refresh = study.start_retained_refresh("series", 2, &initial, None).unwrap();
    assert_eq!(refresh.family(), family);
    let mut model = initial;
    for (mutation, expected) in [
        (
            RefreshMutation::Append { source: source("series"), rows: observations(12..18, false) },
            RefreshOutcome::Updated,
        ),
        (RefreshMutation::SetMask { id: 2, masked: true }, RefreshOutcome::Recomputed),
        (RefreshMutation::Evict { id: 0 }, RefreshOutcome::Recomputed),
    ] {
        assert_eq!(study.apply_retained_refresh(&mut refresh, &mutation).unwrap(), expected);
        apply_to_model(&mut model, &mutation);
        let (intercept, slope) = regression(&model);
        let beta = coefficients(&refresh.result().unwrap());
        assert_close(beta[0], intercept, "intercept");
        assert_close(beta[1], slope, "slope");
    }

    let scratch = Scratch::new("study");
    let (ctx, _) = pooled(1 << 20);
    let registry = spill_registry_within_admission(&ctx, scratch.at("registry"), 1 << 20).unwrap();
    let spilled = study.spill_retained_refresh(&refresh, &ctx, &registry, "study").unwrap();
    assert_eq!(spilled.mode(), RetentionMode::DiskBacked);
    let back = study.reload_retained_refresh(&spilled, &ctx).unwrap();
    assert_eq!(back.result().unwrap(), refresh.result().unwrap());

    // A handle of another family is refused rather than mutated with a mismatched summary.
    let mut scatter =
        RetainedRefresh::start(RefreshFamily::Covariance, "series", 2, &pairs(0..6), None).unwrap();
    let foreign =
        study.apply_retained_refresh(&mut scatter, &RefreshMutation::Evict { id: 0 }).unwrap_err();
    assert!(foreign.to_string().contains("keeps a ols summary"), "{foreign}");
}
