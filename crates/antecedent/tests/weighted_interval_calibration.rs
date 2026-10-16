//! Coverage designs for the normal-approximation intervals of sampling-weighted effects.
//!
//! Units come from a simulated super-population and are included with a known, unequal
//! probability that rises with a confounder `z`; a sampled unit's weight is its inverse
//! inclusion probability. The population effect is `1` (the unit effect is `1 + z/2` and `z`
//! has mean zero), so an estimator that ignored the weights would speak for the over-sampled
//! high-`z` units and miss. Each design scores `effect ± z_{0.975} · se` from the estimator
//! that backs a weighted route, on a sample whose size the grid varies.
//!
//! A weighted interval is another construction than its unweighted namesake: its estimand,
//! its variance and its effective sample size differ, so every record binds the functional
//! label `…+weights=sampling` and no record measured on unweighted rows applies to it. The
//! routes publish no weighted interval until a record measures one. These designs are
//! `#[ignore]`d measurement designs registered in `scripts/gate_calibration.sh`
//! (`run_wi`); the coverage record is the only missing piece. The fixed-seed truth checks of the
//! same estimators are `crates/antecedent-estimate/tests/sampling_weights.rs`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

mod common;

use std::sync::Arc;

use antecedent::{PanelDidQuery, Study};
use antecedent_core::{
    AssumptionSet, AverageEffectQuery, ExecutionContext, StreamDomain, TargetPopulation, VariableId,
};
use antecedent_data::TabularData;
use antecedent_estimate::{
    AipwAte, AipwWorkspace, AnalyticSeKind, ClusterDml, DistanceMatching, DmlAte, DrLearner,
    EffectEstimate, EstimationWorkspace, GlmAdjustmentAte, GlmAdjustmentWorkspace,
    LinearAdjustmentAte, OverlapPolicy, PropensityEstimationWorkspace, PropensityMatching,
    PropensityStratification, PropensityWeighting, SamplingWeights,
};
use antecedent_expr::{ExprId, IdentifiedEstimand};
use antecedent_kernels::{sigmoid, standard_normal};
use antecedent_stats::GlmFamily;
use common::calibration::{
    Construction, CoverageTally, REPORTED_LEVEL, RecordKey, ScopeFacts, Z95, grid_n,
    map_replicates, n_sim,
};

const INTERVAL: &str = "analytic_se";
const TRUTH: f64 = 1.0;
const T: VariableId = VariableId::from_raw(0);
const Y: VariableId = VariableId::from_raw(1);
const Z: VariableId = VariableId::from_raw(2);
/// Rows per cluster of the clustered designs.
const CLUSTER_SIZE: usize = 8;

/// Sampled units with their inverse-inclusion-probability weights and cluster labels.
struct Sample {
    t: Vec<f64>,
    y: Vec<f64>,
    z: Vec<f64>,
    w: Vec<f64>,
    cluster: Vec<u32>,
}

/// How the outcome depends on the treatment.
#[derive(Clone, Copy)]
enum Effect {
    /// `Y = z + T + noise`, noise scale rising in `|z|`: a correctly specified linear model.
    Constant,
    /// `Y = z + (1 + z/2) T + noise`: the population average effect differs from the sampled one.
    Heterogeneous,
}

/// Draw `n` sampled units by Poisson sampling from the super-population. With `clustered`,
/// consecutive sampled units share a cluster effect on the covariate and the outcome.
fn sample(n: usize, seed: u64, effect: Effect, clustered: bool) -> Sample {
    let mut rng = ExecutionContext::for_tests(seed).rng.stream_for(StreamDomain::Test, 0x5A01);
    let mut shocks: Vec<f64> = Vec::new();
    let mut out = Sample { t: vec![], y: vec![], z: vec![], w: vec![], cluster: vec![] };
    while out.t.len() < n {
        let group = if clustered { out.t.len() / CLUSTER_SIZE } else { out.t.len() };
        if clustered && shocks.len() <= group {
            shocks.push(standard_normal(&mut rng));
        }
        let shock = if clustered { shocks[group] } else { 0.0 };
        let z = standard_normal(&mut rng) + 0.5 * shock;
        let t = f64::from(rng.next_f64() < sigmoid(-0.3 + 0.8 * z));
        let noise = standard_normal(&mut rng);
        let y = match effect {
            Effect::Constant => z + t + shock + (0.5 + 0.5 * z.abs()) * noise,
            Effect::Heterogeneous => z + (1.0 + 0.5 * z) * t + shock + 0.7 * noise,
        };
        let inclusion = sigmoid(-1.4 + z);
        if rng.next_f64() < inclusion {
            out.t.push(t);
            out.y.push(y);
            out.z.push(z);
            out.w.push(1.0 / inclusion);
            out.cluster.push(u32::try_from(group).unwrap_or(u32::MAX));
        }
    }
    out
}

fn table(sample: &Sample) -> Option<(TabularData, IdentifiedEstimand)> {
    let data = TabularData::from_f64_columns([
        ("t", sample.t.as_slice()),
        ("y", sample.y.as_slice()),
        ("z", sample.z.as_slice()),
    ])
    .ok()?;
    let estimand =
        IdentifiedEstimand::backdoor("backdoor.adjustment", Arc::from([Z]), ExprId::from_raw(0));
    Some((data, estimand))
}

fn query() -> AverageEffectQuery {
    AverageEffectQuery::binary_ate(T, Y)
}

fn no_clip() -> OverlapPolicy {
    OverlapPolicy::RequireDiagnostics { clip: None, trim: None }
}

/// The scored interval `effect ± z_{0.975} · se`, or `None` when the fit has no usable SE.
fn interval(effect: &EffectEstimate) -> Option<(f64, f64)> {
    let se = effect.se_analytic();
    (effect.ate.is_finite() && se.is_finite() && se > 0.0)
        .then(|| (effect.ate - Z95 * se, effect.ate + Z95 * se))
}

fn linear(sample: &Sample, cluster: bool) -> Option<EffectEstimate> {
    let (data, estimand) = table(sample)?;
    let mut estimator = LinearAdjustmentAte::new()
        .with_bootstrap_replicates(0)
        .with_sampling_weights(SamplingWeights::new(sample.w.clone()).ok()?);
    if cluster {
        estimator = estimator
            .with_se_kind(AnalyticSeKind::Cluster)
            .with_source_cluster_ids(sample.cluster.clone());
    } else {
        estimator = estimator.with_se_kind(AnalyticSeKind::Hc1);
    }
    let prepared = estimator.prepare(&data, &estimand, &query()).ok()?;
    estimator
        .fit(
            &prepared,
            &mut EstimationWorkspace::default(),
            &ExecutionContext::for_tests(1),
            AssumptionSet::new(),
        )
        .ok()
}

fn weighting(sample: &Sample, cluster: bool) -> Option<EffectEstimate> {
    let (data, estimand) = table(sample)?;
    let mut estimator = PropensityWeighting::new()
        .with_bootstrap_replicates(0)
        .with_overlap(no_clip())
        .with_sampling_weights(SamplingWeights::new(sample.w.clone()).ok()?);
    if cluster {
        estimator = estimator.with_source_cluster_ids(sample.cluster.clone());
    }
    let prepared = estimator.prepare(&data, &estimand, &query()).ok()?;
    estimator
        .fit(
            &prepared,
            &mut PropensityEstimationWorkspace::default(),
            &ExecutionContext::for_tests(1),
            AssumptionSet::new(),
        )
        .ok()
}

fn stratification(sample: &Sample) -> Option<EffectEstimate> {
    let (data, estimand) = table(sample)?;
    let estimator = PropensityStratification::new()
        .with_bootstrap_replicates(0)
        .with_overlap(no_clip())
        .with_n_strata(10)
        .with_sampling_weights(SamplingWeights::new(sample.w.clone()).ok()?);
    let prepared = estimator.prepare(&data, &estimand, &query()).ok()?;
    estimator
        .fit(
            &prepared,
            &mut PropensityEstimationWorkspace::default(),
            &ExecutionContext::for_tests(1),
            AssumptionSet::new(),
        )
        .ok()
}

fn matching(sample: &Sample) -> Option<EffectEstimate> {
    let (data, estimand) = table(sample)?;
    let estimator = PropensityMatching::new()
        .with_overlap(no_clip())
        .with_sampling_weights(SamplingWeights::new(sample.w.clone()).ok()?);
    let prepared = estimator.prepare(&data, &estimand, &query()).ok()?;
    estimator
        .fit(
            &prepared,
            &mut PropensityEstimationWorkspace::default(),
            &ExecutionContext::for_tests(1),
            AssumptionSet::new(),
        )
        .ok()
}

fn aipw(sample: &Sample) -> Option<EffectEstimate> {
    let (data, estimand) = table(sample)?;
    let estimator = AipwAte::new()
        .with_bootstrap_replicates(0)
        .with_sampling_weights(SamplingWeights::new(sample.w.clone()).ok()?);
    let prepared = estimator.prepare(&data, &estimand, &query()).ok()?;
    estimator
        .fit(
            &prepared,
            &mut AipwWorkspace::default(),
            &ExecutionContext::for_tests(1),
            AssumptionSet::new(),
        )
        .ok()
}

/// Sampled units whose outcome is binary: `P(Y = 1 | T, z) = σ(-0.3 + 0.8 T + 0.6 z)`, with the
/// same informative sampling and treatment assignment as [`sample`].
fn binary_sample(n: usize, seed: u64) -> Sample {
    let mut rng = ExecutionContext::for_tests(seed).rng.stream_for(StreamDomain::Test, 0x5A04);
    let mut out = Sample { t: vec![], y: vec![], z: vec![], w: vec![], cluster: vec![] };
    while out.t.len() < n {
        let z = standard_normal(&mut rng);
        let t = f64::from(rng.next_f64() < sigmoid(-0.3 + 0.8 * z));
        let y = f64::from(rng.next_f64() < sigmoid(-0.3 + 0.8 * t + 0.6 * z));
        let inclusion = sigmoid(-1.4 + z);
        if rng.next_f64() < inclusion {
            out.t.push(t);
            out.y.push(y);
            out.z.push(z);
            out.w.push(1.0 / inclusion);
            out.cluster.push(u32::try_from(out.t.len()).unwrap_or(u32::MAX));
        }
    }
    out
}

/// The population risk difference of [`binary_sample`]'s outcome model,
/// `E_z[σ(0.5 + 0.6 z) − σ(-0.3 + 0.6 z)]` with `z ~ N(0, 1)`, by a fine midpoint rule.
fn binary_population_effect() -> f64 {
    let (lo, hi, steps) = (-9.0_f64, 9.0_f64, 36_000_usize);
    let width = (hi - lo) / steps as f64;
    (0..steps)
        .map(|i| {
            let z = lo + (i as f64 + 0.5) * width;
            let density = (-0.5 * z * z).exp() / (2.0 * std::f64::consts::PI).sqrt();
            density * width * (sigmoid(0.5 + 0.6 * z) - sigmoid(-0.3 + 0.6 * z))
        })
        .sum()
}

fn glm(sample: &Sample, family: GlmFamily) -> Option<EffectEstimate> {
    let (data, estimand) = table(sample)?;
    let estimator = GlmAdjustmentAte::new()
        .with_bootstrap_replicates(0)
        .with_family(family)
        .with_se_kind(AnalyticSeKind::Hc1)
        .with_sampling_weights(SamplingWeights::new(sample.w.clone()).ok()?);
    let prepared = estimator.prepare(&data, &estimand, &query()).ok()?;
    estimator
        .fit(
            &prepared,
            &mut GlmAdjustmentWorkspace::default(),
            &ExecutionContext::for_tests(1),
            AssumptionSet::new(),
        )
        .ok()
}

fn distance_matching(sample: &Sample) -> Option<EffectEstimate> {
    let (data, estimand) = table(sample)?;
    let estimator = DistanceMatching::new()
        .with_overlap(no_clip())
        .with_sampling_weights(SamplingWeights::new(sample.w.clone()).ok()?);
    let prepared = estimator.prepare(&data, &estimand, &query()).ok()?;
    estimator
        .fit(
            &prepared,
            &mut PropensityEstimationWorkspace::default(),
            &ExecutionContext::for_tests(1),
            AssumptionSet::new(),
        )
        .ok()
}

fn dml(sample: &Sample, partially_linear: bool) -> Option<EffectEstimate> {
    let (data, estimand) = table(sample)?;
    let mut estimator =
        DmlAte::new().with_sampling_weights(SamplingWeights::new(sample.w.clone()).ok()?);
    if partially_linear {
        estimator = estimator.with_score(antecedent_estimate::DmlScore::PartiallyLinear);
    }
    let prepared = estimator.prepare(&data, &estimand, &query()).ok()?;
    estimator.fit(&prepared, &ExecutionContext::for_tests(1), AssumptionSet::new()).ok()
}

fn dr_learner(sample: &Sample) -> Option<EffectEstimate> {
    let (data, estimand) = table(sample)?;
    let estimator =
        DrLearner::new().with_sampling_weights(SamplingWeights::new(sample.w.clone()).ok()?);
    let prepared = estimator.prepare(&data, &estimand, &query()).ok()?;
    estimator.fit(&prepared, &ExecutionContext::for_tests(1), AssumptionSet::new()).ok()
}

/// AIPW on the in-sample route: a treated or untreated target, or a trimmed ATE.
fn aipw_in_sample(
    sample: &Sample,
    target: TargetPopulation,
    overlap: OverlapPolicy,
) -> Option<EffectEstimate> {
    let (data, estimand) = table(sample)?;
    let estimator = AipwAte::new()
        .with_bootstrap_replicates(0)
        .with_overlap(overlap)
        .with_sampling_weights(SamplingWeights::new(sample.w.clone()).ok()?);
    let prepared =
        estimator.prepare(&data, &estimand, &query().with_target_population(target)).ok()?;
    estimator
        .fit(
            &prepared,
            &mut AipwWorkspace::default(),
            &ExecutionContext::for_tests(1),
            AssumptionSet::new(),
        )
        .ok()
}

/// Cross-fitted AIPW with whole-cluster folds and the weighted cluster sandwich.
fn aipw_cluster_dml(sample: &Sample) -> Option<EffectEstimate> {
    let (data, estimand) = table(sample)?;
    let estimator = AipwAte::new()
        .with_bootstrap_replicates(0)
        .with_sampling_weights(SamplingWeights::new(sample.w.clone()).ok()?)
        .with_cluster_dml(ClusterDml::new(20).ok()?)
        .with_source_cluster_ids(sample.cluster.clone());
    let prepared = estimator.prepare(&data, &estimand, &query()).ok()?;
    estimator
        .fit(
            &prepared,
            &mut AipwWorkspace::default(),
            &ExecutionContext::for_tests(1),
            AssumptionSet::new(),
        )
        .ok()
}

/// Bernoulli-randomized trial on sampled units: assignment is independent of everything with
/// probability one half, so the weights are the only thing separating the sample from the
/// population.
fn randomized(sample: &Sample, seed: u64) -> Option<((f64, f64), u64)> {
    let mut rng = ExecutionContext::for_tests(seed).rng.stream_for(StreamDomain::Test, 0x5A02);
    let n = sample.t.len();
    let assigned: Vec<bool> = (0..n).map(|_| rng.next_f64() < 0.5).collect();
    let outcome: Vec<f64> = (0..n)
        .map(|i| {
            let z = sample.z[i];
            z + if assigned[i] { 1.0 + 0.5 * z } else { 0.0 } + 0.7 * standard_normal(&mut rng)
        })
        .collect();
    let fit = antecedent_estimate::randomized_neyman::weighted_itt(
        &outcome,
        &assigned,
        &vec![0.5; n],
        &sample.w,
    )?;
    let radius = Z95 * fit.variance.sqrt();
    Some(((fit.effect - radius, fit.effect + radius), n as u64))
}

/// Balanced two-period panel: cluster-level treatment, a baseline effect of `z`, and a
/// post-period change that carries the heterogeneous effect.
fn panel(clusters_per_arm: usize, seed: u64) -> Option<((f64, f64), u64)> {
    let mut rng = ExecutionContext::for_tests(seed).rng.stream_for(StreamDomain::Test, 0x5A03);
    let (mut subjects, mut clusters, mut treated, mut post) = (vec![], vec![], vec![], vec![]);
    let (mut outcome, mut weights) = (vec![], vec![]);
    let mut subject = 0_usize;
    for cluster in 0..2 * clusters_per_arm {
        let active = cluster >= clusters_per_arm;
        let shock = 0.6 * standard_normal(&mut rng);
        let mut members = 0;
        while members < 4 {
            let z = standard_normal(&mut rng) + 0.5 * shock;
            let inclusion = sigmoid(-1.4 + z);
            if rng.next_f64() >= inclusion {
                continue;
            }
            let weight = 1.0 / inclusion;
            let baseline = z + 0.5 * standard_normal(&mut rng);
            let effect = if active { 1.0 + 0.5 * z } else { 0.0 };
            let change = 0.3 + effect + shock + 0.5 * standard_normal(&mut rng);
            for (after, value) in [(false, baseline), (true, baseline + change)] {
                subjects.push(Arc::<str>::from(format!("s{subject}")));
                clusters.push(Arc::<str>::from(format!("c{cluster}")));
                treated.push(active);
                post.push(after);
                outcome.push(value);
                weights.push(weight);
            }
            subject += 1;
            members += 1;
        }
    }
    let n = outcome.len();
    let data = TabularData::from_f64_columns([("outcome", outcome.as_slice())])
        .ok()?
        .with_weights(Arc::from(weights))
        .ok()?;
    let query = PanelDidQuery::new(VariableId::from_raw(0), treated, post, subjects, clusters);
    let result = Study::tabular(data)
        .query(query)
        .build()
        .ok()?
        .run(&ExecutionContext::for_tests(seed))
        .ok()?;
    let fit = result.panel_did()?;
    let radius = Z95 * fit.standard_error;
    Some(((fit.effect - radius, fit.effect + radius), n as u64))
}

fn construction(estimator: &str, se_kind: &str, dependence: &str, query: &str) -> Construction {
    construction_for(estimator, se_kind, dependence, query, "all_observed.mean")
}

/// [`construction`] for a functional other than the average over all rows: `functional` is the
/// unweighted label (population, outcome functional and any overlap policy), to which the
/// sampling-weights suffix is added.
fn construction_for(
    estimator: &str,
    se_kind: &str,
    dependence: &str,
    query: &str,
    functional: &str,
) -> Construction {
    Construction {
        query: query.into(),
        graph_class: if query == "AverageEffect" { "Dag" } else { "RandomizedTrial" }.into(),
        structure: if query == "AverageEffect" { "fixed" } else { "design" }.into(),
        modality: "tabular".into(),
        inference: "Frequentist".into(),
        estimator: estimator.into(),
        interval_method: INTERVAL.into(),
        se_kind: se_kind.into(),
        dependence: dependence.into(),
        posterior: String::new(),
        functional: format!("{functional}+{}", antecedent::estimator_spec::SAMPLING_WEIGHTS_LABEL),
        identification: "point".into(),
        reported_level: REPORTED_LEVEL,
    }
}

fn measure(
    test: &'static str,
    dgp: &'static str,
    base: usize,
    construction: &Construction,
    replicate: impl Fn(usize, u64) -> Option<((f64, f64), u64)> + Sync,
) {
    measure_against(TRUTH, test, dgp, base, construction, replicate);
}

fn measure_against(
    truth: f64,
    test: &'static str,
    dgp: &'static str,
    base: usize,
    construction: &Construction,
    replicate: impl Fn(usize, u64) -> Option<((f64, f64), u64)> + Sync,
) {
    let n = grid_n(base);
    let mut tally =
        CoverageTally::for_record(RecordKey { test, dgp, interval: INTERVAL }, REPORTED_LEVEL);
    for row in map_replicates(n_sim(), |rep| replicate(n, 91_000 + rep)) {
        let Some((interval, rows)) = row else {
            tally.skip();
            continue;
        };
        tally.bind(
            construction,
            ScopeFacts {
                row_count: rows,
                replicates_ok: None,
                posterior_draws: None,
                unidentified_mass: 0.0,
            },
        );
        tally.record(Some(interval), truth);
    }
    tally.assert();
}

fn scored(fit: Option<EffectEstimate>, rows: usize) -> Option<((f64, f64), u64)> {
    Some((interval(&fit?)?, rows as u64))
}

/// Every scored interval is finite, ordered and centred near the population effect.
#[test]
fn the_scored_intervals_are_ordered_and_centred_near_the_population_effect() {
    let heterogeneous = sample(3_000, 91_001, Effect::Heterogeneous, false);
    let constant = sample(3_000, 91_002, Effect::Constant, false);
    let clustered = sample(3_000, 91_003, Effect::Heterogeneous, true);
    let fits = [
        scored(linear(&constant, false), 3_000),
        scored(linear(&clustered, true), 3_000),
        scored(weighting(&heterogeneous, false), 3_000),
        scored(weighting(&clustered, true), 3_000),
        scored(aipw(&heterogeneous), 3_000),
        scored(stratification(&heterogeneous), 3_000),
        scored(matching(&heterogeneous), 3_000),
        randomized(&heterogeneous, 91_004),
        scored(distance_matching(&heterogeneous), 3_000),
        scored(dml(&heterogeneous, false), 3_000),
        scored(dr_learner(&heterogeneous), 3_000),
        scored(glm(&constant, GlmFamily::GaussianIdentity), 3_000),
        scored(aipw_in_sample(&constant, TargetPopulation::Treated, no_clip()), 3_000),
        scored(
            aipw_in_sample(
                &constant,
                TargetPopulation::AllObserved,
                OverlapPolicy::RequireDiagnostics { clip: None, trim: Some(0.05) },
            ),
            3_000,
        ),
        scored(aipw_cluster_dml(&clustered), 3_000),
    ];
    for fit in fits {
        let ((lo, hi), _) = fit.expect("a weighted fit");
        assert!(lo.is_finite() && hi.is_finite() && lo < hi);
        assert!(((lo + hi) / 2.0 - TRUTH).abs() < 0.4, "centre {}", (lo + hi) / 2.0);
    }
    let ((lo, hi), rows) = panel(40, 91_005).expect("a weighted panel fit");
    assert_eq!(rows, 640);
    assert!(lo < hi && ((lo + hi) / 2.0 - TRUTH).abs() < 0.6);
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn weighted_linear_adjustment_hc1_interval() {
    measure(
        "weighted_linear_adjustment_hc1_interval",
        "informative_sampling_constant_effect",
        1200,
        &construction("linear.adjustment.ate", "hc1", "iid", "AverageEffect"),
        |n, seed| {
            let s = sample(n, seed, Effect::Constant, false);
            scored(linear(&s, false), n)
        },
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn weighted_linear_adjustment_cluster_interval() {
    measure(
        "weighted_linear_adjustment_cluster_interval",
        "informative_sampling_clustered",
        1200,
        &construction("linear.adjustment.ate", "cluster", "cluster", "AverageEffect"),
        |n, seed| {
            let s = sample(n, seed, Effect::Constant, true);
            scored(linear(&s, true), n)
        },
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn weighted_ipw_hajek_interval() {
    measure(
        "weighted_ipw_hajek_interval",
        "informative_sampling_heterogeneous_effect",
        1200,
        &construction("propensity.weighting", "", "iid", "AverageEffect"),
        |n, seed| {
            let s = sample(n, seed, Effect::Heterogeneous, false);
            scored(weighting(&s, false), n)
        },
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn weighted_ipw_hajek_cluster_interval() {
    measure(
        "weighted_ipw_hajek_cluster_interval",
        "informative_sampling_clustered_heterogeneous_effect",
        1200,
        &construction("propensity.weighting", "cluster", "cluster", "AverageEffect"),
        |n, seed| {
            let s = sample(n, seed, Effect::Heterogeneous, true);
            scored(weighting(&s, true), n)
        },
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn weighted_aipw_interval() {
    measure(
        "weighted_aipw_interval",
        "informative_sampling_heterogeneous_effect",
        1200,
        &construction("aipw", "homoskedastic", "iid", "AverageEffect"),
        |n, seed| {
            let s = sample(n, seed, Effect::Heterogeneous, false);
            scored(aipw(&s), n)
        },
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn weighted_stratification_interval() {
    measure(
        "weighted_stratification_interval",
        "informative_sampling_heterogeneous_effect",
        1200,
        &construction("propensity.stratification", "", "iid", "AverageEffect"),
        |n, seed| {
            let s = sample(n, seed, Effect::Heterogeneous, false);
            scored(stratification(&s), n)
        },
    );
}

/// Weighted matching earns its own record: the variance of a weighted matched mean is another
/// construction than the equal-weight Abadie-Imbens one (`matching_homoskedastic_ci_coverage`).
#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn weighted_matching_interval() {
    measure(
        "weighted_matching_interval",
        "informative_sampling_heterogeneous_effect",
        1200,
        &construction("propensity.matching", "homoskedastic", "iid", "AverageEffect"),
        |n, seed| {
            let s = sample(n, seed, Effect::Heterogeneous, false);
            scored(matching(&s), n)
        },
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn weighted_randomized_itt_interval() {
    measure(
        "weighted_randomized_itt_interval",
        "bernoulli_trial_on_informatively_sampled_units",
        600,
        &construction("randomized_neyman", "", "iid", "RandomizedEffect"),
        |n, seed| {
            let s = sample(n, seed, Effect::Heterogeneous, false);
            randomized(&s, seed)
        },
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn weighted_panel_did_interval() {
    measure(
        "weighted_panel_did_interval",
        "cluster_panel_informative_sampling",
        60,
        &construction("randomized_ht", "cluster", "cluster", "PanelDid"),
        |clusters_per_arm, seed| panel(clusters_per_arm, seed),
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn weighted_glm_logit_interval() {
    measure_against(
        binary_population_effect(),
        "weighted_glm_logit_interval",
        "informative_sampling_binary_outcome",
        1200,
        &construction("glm.adjustment", "hc1", "iid", "AverageEffect"),
        |n, seed| {
            let s = binary_sample(n, seed);
            scored(glm(&s, GlmFamily::BinomialLogit), n)
        },
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn weighted_glm_gaussian_interval() {
    measure(
        "weighted_glm_gaussian_interval",
        "informative_sampling_constant_effect",
        1200,
        &construction("glm.adjustment", "hc1", "iid", "AverageEffect"),
        |n, seed| {
            let s = sample(n, seed, Effect::Constant, false);
            scored(glm(&s, GlmFamily::GaussianIdentity), n)
        },
    );
}

/// Distance matching earns its own weighted record, apart from propensity-score matching: the
/// metric is the standardized covariate distance and the diagnostic propensity fit is weighted.
#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn weighted_distance_matching_interval() {
    measure(
        "weighted_distance_matching_interval",
        "informative_sampling_heterogeneous_effect",
        1200,
        &construction("distance.matching", "homoskedastic", "iid", "AverageEffect"),
        |n, seed| {
            let s = sample(n, seed, Effect::Heterogeneous, false);
            scored(distance_matching(&s), n)
        },
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn weighted_dml_aipw_interval() {
    measure(
        "weighted_dml_aipw_interval",
        "informative_sampling_heterogeneous_effect",
        1200,
        &construction("dml", "homoskedastic", "iid", "AverageEffect"),
        |n, seed| {
            let s = sample(n, seed, Effect::Heterogeneous, false);
            scored(dml(&s, false), n)
        },
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn weighted_dml_partially_linear_interval() {
    measure(
        "weighted_dml_partially_linear_interval",
        "informative_sampling_constant_effect",
        1200,
        &construction("dml", "homoskedastic", "iid", "AverageEffect"),
        |n, seed| {
            let s = sample(n, seed, Effect::Constant, false);
            scored(dml(&s, true), n)
        },
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn weighted_dr_learner_interval() {
    measure(
        "weighted_dr_learner_interval",
        "informative_sampling_heterogeneous_effect",
        1200,
        &construction("dr.learner", "homoskedastic", "iid", "AverageEffect"),
        |n, seed| {
            let s = sample(n, seed, Effect::Heterogeneous, false);
            scored(dr_learner(&s), n)
        },
    );
}

/// ATT on a constant-effect design: the treated-population effect equals the average effect,
/// so the truth is `1` while the in-sample weighted score, arm share and nuisance correction
/// are all exercised.
#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn weighted_aipw_att_interval() {
    measure(
        "weighted_aipw_att_interval",
        "informative_sampling_constant_effect",
        1200,
        &construction_for(
            "aipw",
            "homoskedastic",
            "iid",
            "AverageEffect",
            "treated.mean+overlap=clip:none,trim:none",
        ),
        |n, seed| {
            let s = sample(n, seed, Effect::Constant, false);
            scored(aipw_in_sample(&s, TargetPopulation::Treated, no_clip()), n)
        },
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn weighted_aipw_atc_interval() {
    measure(
        "weighted_aipw_atc_interval",
        "informative_sampling_constant_effect",
        1200,
        &construction_for(
            "aipw",
            "homoskedastic",
            "iid",
            "AverageEffect",
            "untreated.mean+overlap=clip:none,trim:none",
        ),
        |n, seed| {
            let s = sample(n, seed, Effect::Constant, false);
            scored(aipw_in_sample(&s, TargetPopulation::Untreated, no_clip()), n)
        },
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn weighted_aipw_trimmed_interval() {
    measure(
        "weighted_aipw_trimmed_interval",
        "informative_sampling_constant_effect",
        1200,
        &construction_for(
            "aipw",
            "homoskedastic",
            "iid",
            "AverageEffect",
            "all_observed.mean+overlap=clip:none,trim:0.05",
        ),
        |n, seed| {
            let s = sample(n, seed, Effect::Constant, false);
            scored(
                aipw_in_sample(
                    &s,
                    TargetPopulation::AllObserved,
                    OverlapPolicy::RequireDiagnostics { clip: None, trim: Some(0.05) },
                ),
                n,
            )
        },
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn weighted_aipw_cluster_dml_interval() {
    measure(
        "weighted_aipw_cluster_dml_interval",
        "informative_sampling_clustered_heterogeneous_effect",
        1200,
        &construction("aipw", "cluster", "cluster", "AverageEffect"),
        |n, seed| {
            let s = sample(n, seed, Effect::Heterogeneous, true);
            scored(aipw_cluster_dml(&s), n)
        },
    );
}
