//! Coverage designs for declared sampling-weighted effect intervals.
//!
//! Units come from a simulated super-population and are included with a known, unequal
//! probability that rises with a confounder `z`; a sampled unit's weight is its inverse
//! inclusion probability. The population effect is `1` (the unit effect is `1 + z/2` and `z`
//! has mean zero), so an estimator that ignored the weights would speak for the over-sampled
//! high-`z` units and miss. Designs score the declared public endpoints: cluster-DML
//! uses its published Student reference; IID balanced IPW and cross-fitted population-linear
//! methods use model-assisted score intervals. Other routes
//! score `effect ± z_{0.975} · se`. Stratification declares linear regression adjustment
//! within finite propensity bins. Every construction is measured on the same original
//! sample-size grid and super-population truth.
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
    AssumptionSet, AverageEffectQuery, CausalRng, ExecutionContext, StreamDomain, TargetPopulation,
    VariableId,
};
use antecedent_data::TabularData;
use antecedent_estimate::{
    AipwAte, AipwWorkspace, AnalyticSeKind, ClusterDml, DistanceMatching, DmlAte, DrLearner,
    EffectEstimate, EstimationWorkspace, GlmAdjustmentAte, GlmAdjustmentWorkspace,
    LinearAdjustmentAte, OverlapPolicy, PropensityEstimationWorkspace, PropensityMatching,
    PropensityNuisance, PropensityStratification, PropensityWeighting, RidgeTuning,
    SamplingWeights,
};
use antecedent_expr::{ExprId, IdentifiedEstimand};
use antecedent_kernels::{sigmoid, standard_normal};
use antecedent_stats::GlmFamily;
use common::calibration::{
    Construction, CoverageTally, REPORTED_LEVEL, RecordKey, ScopeFacts, Z95, grid_n,
    map_replicates, n_sim, validation_seed,
};

const INTERVAL: &str = "analytic_se";
const TRUTH: f64 = 1.0;
const T: VariableId = VariableId::from_raw(0);
const Y: VariableId = VariableId::from_raw(1);
const Z: VariableId = VariableId::from_raw(2);
/// Population units per cluster of the clustered designs (about eight are sampled).
const CLUSTER_POPULATION: usize = 32;

/// Population subjects per cluster of the panel design (about four are sampled).
const PANEL_CLUSTER_POPULATION: usize = 16;

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

/// One super-population unit: its covariate, treatment, outcome and inclusion probability. The
/// covariate carries half of its cluster's `shock`, which also enters the outcome.
fn population_unit(rng: &mut CausalRng, shock: f64, effect: Effect) -> (f64, f64, f64, f64) {
    let z = standard_normal(rng) + 0.5 * shock;
    let t = f64::from(rng.next_f64() < sigmoid(-0.3 + 0.8 * z));
    let noise = standard_normal(rng);
    let y = match effect {
        Effect::Constant => z + t + shock + (0.5 + 0.5 * z.abs()) * noise,
        Effect::Heterogeneous => z + (1.0 + 0.5 * z) * t + shock + 0.7 * noise,
    };
    (z, t, y, sigmoid(-1.4 + z))
}

/// Draw `n` sampled units by Poisson sampling from the super-population. With `clustered`, the
/// population is made of clusters of [`CLUSTER_POPULATION`] units sharing a cluster effect on
/// the covariate and the outcome, and each unit of a cluster is included independently, so a
/// cluster's sampled size varies with its effect and the weights represent the population the
/// effect and the truth are defined on. (Stopping each cluster after a fixed number of
/// inclusions would give a cluster with a low inclusion rate more population units than one
/// with a high rate, which tilts the weighted population away from the one the truth speaks
/// for.) The final cluster is cut at `n` rows.
fn sample(n: usize, seed: u64, effect: Effect, clustered: bool) -> Sample {
    let mut rng = ExecutionContext::for_tests(validation_seed(seed))
        .rng
        .stream_for(StreamDomain::Test, 0x5A01);
    let mut out = Sample { t: vec![], y: vec![], z: vec![], w: vec![], cluster: vec![] };
    let mut group = 0_u32;
    while out.t.len() < n {
        let (shock, units) =
            if clustered { (standard_normal(&mut rng), CLUSTER_POPULATION) } else { (0.0, 1) };
        for _ in 0..units {
            let (z, t, y, inclusion) = population_unit(&mut rng, shock, effect);
            if rng.next_f64() < inclusion {
                out.t.push(t);
                out.y.push(y);
                out.z.push(z);
                out.w.push(1.0 / inclusion);
                out.cluster.push(group);
            }
        }
        group += 1;
    }
    for column in [&mut out.t, &mut out.y, &mut out.z, &mut out.w] {
        column.truncate(n);
    }
    out.cluster.truncate(n);
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
        .then_some((effect.ate - Z95 * se, effect.ate + Z95 * se))
}

/// The historical normal-reference calibration is replay-only. Ordinary
/// measurements score the published finite-cluster Student reference.
fn cluster_protocol_version() -> u32 {
    static VERSION: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *VERSION.get_or_init(|| {
        let raw =
            std::env::var("ANTECEDENT_WEIGHTED_CLUSTER_PROTOCOL").unwrap_or_else(|_| "2".into());
        raw.parse::<u32>()
            .ok()
            .filter(|version| matches!(version, 1 | 2))
            .unwrap_or_else(|| panic!("ANTECEDENT_WEIGHTED_CLUSTER_PROTOCOL={raw}: want 1 or 2"))
    })
}

/// Score the actual finite-cluster reference published by cluster-DML estimators.
fn scored_cluster_reference(fit: Option<EffectEstimate>, rows: usize) -> Option<((f64, f64), u64)> {
    scored_cluster_reference_for_protocol(fit, rows, cluster_protocol_version())
}

fn scored_cluster_reference_for_protocol(
    fit: Option<EffectEstimate>,
    rows: usize,
    protocol: u32,
) -> Option<((f64, f64), u64)> {
    if protocol == 1 {
        return scored(fit, rows);
    }
    let fit = fit?;
    let df = fit.uncertainty.analytic.reference_df?;
    let critical = antecedent_stats::student_t_ppf((1.0 + REPORTED_LEVEL) / 2.0, df);
    let se = fit.se_analytic();
    (df.is_finite()
        && df > 0.0
        && critical.is_finite()
        && se.is_finite()
        && se > 0.0
        && fit.ate.is_finite())
    .then_some(((fit.ate - critical * se, fit.ate + critical * se), rows as u64))
}

fn cluster_reference_construction(estimator: &str) -> Construction {
    let protocol = cluster_protocol_version();
    eprintln!("calibration-protocol weighted_cluster_source_version={protocol}");
    cluster_reference_construction_for_protocol(estimator, protocol)
}

fn cluster_reference_construction_for_protocol(estimator: &str, protocol: u32) -> Construction {
    if protocol == 1 {
        return construction(estimator, "cluster", "cluster", "AverageEffect");
    }
    let mut basis = construction_for(
        estimator,
        "cluster",
        "cluster",
        "AverageEffect",
        "all_observed.mean+reference=cluster_df",
    );
    basis.interval_method = "student_analytic".into();
    basis
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
        .with_covariate_balance(true)
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
        .with_regression_adjustment(true)
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
        .with_population_variance(true)
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
    let mut rng = ExecutionContext::for_tests(validation_seed(seed))
        .rng
        .stream_for(StreamDomain::Test, 0x5A04);
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
        .with_population_variance(true)
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
    let mut rng = ExecutionContext::for_tests(validation_seed(seed))
        .rng
        .stream_for(StreamDomain::Test, 0x5A02);
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

/// Two-period panel: cluster-level treatment, a baseline effect of `z`, and a post-period change
/// that carries the heterogeneous effect. Each cluster holds [`PANEL_CLUSTER_POPULATION`]
/// population subjects, each included independently, so the weights represent a fixed
/// population and the truth is its effect (a fixed number of inclusions per cluster would give
/// low-inclusion clusters more population subjects).
fn panel(clusters_per_arm: usize, seed: u64) -> Option<((f64, f64), u64)> {
    let mut rng = ExecutionContext::for_tests(validation_seed(seed))
        .rng
        .stream_for(StreamDomain::Test, 0x5A03);
    let (mut subjects, mut clusters, mut treated, mut post) = (vec![], vec![], vec![], vec![]);
    let (mut outcome, mut weights) = (vec![], vec![]);
    let mut subject = 0_usize;
    for cluster in 0..2 * clusters_per_arm {
        let active = cluster >= clusters_per_arm;
        let shock = 0.6 * standard_normal(&mut rng);
        for _ in 0..PANEL_CLUSTER_POPULATION {
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
        .run(&ExecutionContext::for_tests(validation_seed(seed)))
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
    let mut tally = analytic_tally(test, dgp, construction);
    let mut centres = Vec::new();
    let mut se_squared_sum = 0.0;
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
        centres.push((interval.0 + interval.1) / 2.0);
        se_squared_sum += ((interval.1 - interval.0) / (2.0 * Z95)).powi(2);
    }
    if centres.len() > 1 {
        let count = centres.len() as f64;
        let mean = centres.iter().sum::<f64>() / count;
        let sd = (centres.iter().map(|centre| (centre - mean).powi(2)).sum::<f64>()
            / (count - 1.0))
            .sqrt();
        eprintln!(
            "info interval diagnostics {test}: bias={:.5} empirical_sd={sd:.5} rms_se={:.5}",
            mean - truth,
            (se_squared_sum / count).sqrt(),
        );
    }
    tally.assert();
}

/// Preserve collected raw-construction records when a gate measures its declared upgrade.
fn analytic_tally(test: &'static str, dgp: &'static str, basis: &Construction) -> CoverageTally {
    let interval = match basis.interval_method.as_str() {
        "student_analytic" => "student_analytic",
        _ => INTERVAL,
    };
    let tally = CoverageTally::for_record(RecordKey { test, dgp, interval }, REPORTED_LEVEL);
    if basis.functional.contains("matching=population_linear") {
        tally.labelled("population_linear")
    } else if basis.functional.contains("stratification=regression_linear_hc3") {
        tally.labelled("regression_linear_hc3")
    } else {
        tally
    }
}

#[test]
fn upgraded_weighted_constructions_preserve_legacy_record_identity() {
    for (estimator, test, functional, suffix) in [
        (
            "propensity.matching",
            "weighted_matching_interval",
            "all_observed.mean+matching=population_linear",
            "population_linear",
        ),
        (
            "distance.matching",
            "weighted_distance_matching_interval",
            "all_observed.mean+matching=population_linear",
            "population_linear",
        ),
        (
            "propensity.stratification",
            "weighted_stratification_interval",
            "all_observed.mean+stratification=regression_linear_hc3",
            "regression_linear_hc3",
        ),
    ] {
        let raw = construction(estimator, "homoskedastic", "iid", "AverageEffect");
        let upgraded =
            construction_for(estimator, "homoskedastic", "iid", "AverageEffect", functional);
        let mut old = analytic_tally(test, "identity_probe", &raw);
        let mut new = analytic_tally(test, "identity_probe", &upgraded);
        let facts = ScopeFacts {
            row_count: 1200,
            replicates_ok: None,
            posterior_draws: None,
            unidentified_mass: 0.0,
        };
        old.bind(&raw, facts);
        new.bind(&upgraded, facts);
        let old_id = old.record_id().unwrap();
        let new_id = new.record_id().unwrap();
        assert_ne!(old_id, new_id, "an upgraded method must preserve the collected raw record");
        assert_eq!(new_id, format!("{old_id}.{suffix}"));
    }
    let mut basis = construction_for(
        "propensity.weighting",
        "",
        "iid",
        "AverageEffect",
        "all_observed.mean+weighting=balanced_linear_hc3",
    );
    let mut ids = Vec::new();
    for method in ["balanced_population_score", "balanced_population_linear_score"] {
        basis.interval_method = method.into();
        let mut tally = CoverageTally::for_record(
            RecordKey {
                test: "weighted_ipw_hajek_interval",
                dgp: "identity_probe",
                interval: method,
            },
            REPORTED_LEVEL,
        );
        tally.bind(
            &basis,
            ScopeFacts {
                row_count: 1200,
                replicates_ok: None,
                posterior_draws: None,
                unidentified_mass: 0.0,
            },
        );
        ids.push(tally.record_id().unwrap());
    }
    assert_ne!(ids[0], ids[1], "the new model variance must preserve old Satterthwaite records");
}

#[test]
fn cluster_scoring_uses_the_published_student_reference_and_preserves_normal_evidence() {
    let mut fit = EffectEstimate::new(2.0, 0.5, AssumptionSet::new(), no_clip());
    fit.uncertainty.analytic.reference_df = Some(4.0);
    let normal = interval(&fit).unwrap();
    let student = scored_cluster_reference_for_protocol(Some(fit), 17, 2).unwrap();
    let critical = antecedent_stats::student_t_ppf(0.975, 4.0);
    assert_eq!(student.1, 17);
    assert!((student.0.0 - (2.0 - critical * 0.5)).abs() < 1e-12);
    assert!(student.0.0 < normal.0 && student.0.1 > normal.1);
    let raw = construction("aipw", "cluster", "cluster", "AverageEffect");
    let upgraded = cluster_reference_construction_for_protocol("aipw", 2);
    let mut ids = Vec::new();
    for basis in [raw, upgraded] {
        let mut tally =
            analytic_tally("weighted_aipw_cluster_dml_interval", "identity_probe", &basis);
        tally.bind(
            &basis,
            ScopeFacts {
                row_count: 1200,
                replicates_ok: None,
                posterior_draws: None,
                unidentified_mass: 0.0,
            },
        );
        ids.push(tally.record_id().unwrap());
    }
    assert_ne!(ids[0], ids[1]);
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
        // At these seeds every centre is within 0.08 of the population effect and every
        // interval holds it. Ignoring the weights speaks for the over-sampled high-z units and
        // lands near 1.2, outside this band.
        let centre = (lo + hi) / 2.0;
        assert!((centre - TRUTH).abs() < 0.12, "centre {centre}");
        assert!(lo < TRUTH && TRUTH < hi, "interval [{lo}, {hi}] misses the population effect");
    }
    let ((lo, hi), rows) = panel(40, 91_005).expect("a weighted panel fit");
    assert!(rows > 320 && rows % 2 == 0);
    assert!(lo < hi && ((lo + hi) / 2.0 - TRUTH).abs() < 0.15);
    assert!(lo < TRUTH && TRUTH < hi, "panel interval [{lo}, {hi}] misses the population effect");
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
    measure_balanced_hajek("weighted_ipw_hajek_interval", false);
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn weighted_ipw_hajek_cluster_interval() {
    measure_balanced_hajek("weighted_ipw_hajek_cluster_interval", true);
}

fn measure_balanced_hajek(test: &'static str, cluster: bool) {
    let method =
        if cluster { "balanced_population_score" } else { "balanced_population_linear_score" };
    let n = grid_n(1200);
    let dgp = if cluster {
        "informative_sampling_clustered_heterogeneous_effect"
    } else {
        "informative_sampling_heterogeneous_effect"
    };
    let mut basis = construction_for(
        "propensity.weighting",
        if cluster { "cluster" } else { "population_linear_homoskedastic" },
        if cluster { "cluster" } else { "iid" },
        "AverageEffect",
        "all_observed.mean+weighting=balanced_linear_hc3",
    );
    basis.interval_method = method.into();
    if !cluster {
        basis
            .functional
            .push_str("+interval_variance=homoskedastic_gaussian_arm_linear_population");
    }
    let mut tally =
        CoverageTally::for_record(RecordKey { test, dgp, interval: method }, REPORTED_LEVEL);
    let mut reference_df = Vec::new();
    for fit in map_replicates(n_sim(), |rep| {
        let sample = sample(n, 91_000 + rep, Effect::Heterogeneous, cluster);
        let (data, estimand) = table(&sample)?;
        let mut estimator = PropensityWeighting::new()
            .with_bootstrap_replicates(0)
            .with_overlap(no_clip())
            .with_sampling_weights(SamplingWeights::new(sample.w).ok()?)
            .with_covariate_balance(true);
        if cluster {
            estimator = estimator.with_source_cluster_ids(sample.cluster);
        }
        let prepared = estimator.prepare(&data, &estimand, &query()).ok()?;
        let mut workspace = PropensityEstimationWorkspace::default();
        let context = ExecutionContext::for_tests(1);
        if cluster {
            estimator
                .population_score_interval(
                    &prepared,
                    &mut workspace,
                    &context,
                    AssumptionSet::new(),
                    REPORTED_LEVEL,
                )
                .ok()
        } else {
            estimator
                .population_linear_score_interval(
                    &prepared,
                    &mut workspace,
                    &context,
                    AssumptionSet::new(),
                    REPORTED_LEVEL,
                )
                .ok()
        }
    }) {
        let Some(fit) = fit else {
            tally.skip();
            continue;
        };
        if let Some(df) = fit.reference_df {
            reference_df.push(df);
        }
        tally.bind(
            &basis,
            ScopeFacts {
                row_count: n as u64,
                replicates_ok: None,
                posterior_draws: None,
                unidentified_mass: 0.0,
            },
        );
        tally.record(Some((fit.lower, fit.upper)), TRUTH);
    }
    if !reference_df.is_empty() {
        let mean = reference_df.iter().sum::<f64>() / reference_df.len() as f64;
        eprintln!(
            "info calibrated interval reference df {test}: mean={mean:.3} min={:.3} max={:.3}",
            reference_df.iter().copied().fold(f64::INFINITY, f64::min),
            reference_df.iter().copied().fold(0.0_f64, f64::max)
        );
    }
    tally.assert();
}

/// Score the actual public model-assisted endpoints; no normal reconstruction in the harness.
fn measure_population_aipw(
    test: &'static str,
    estimator: &str,
    fit: impl Fn(&Sample) -> Option<antecedent_estimate::population_aipw::PopulationAipwInterval> + Sync,
) {
    let n = grid_n(1200);
    let method = "population_linear_aipw_score";
    let mut basis = construction_for(
        estimator,
        "population_linear_homoskedastic",
        "iid",
        "AverageEffect",
        "all_observed.mean+interval_variance=homoskedastic_gaussian_arm_linear_population+point_correction=arm_linear_projection",
    );
    basis.interval_method = method.into();
    let mut tally = CoverageTally::for_record(
        RecordKey { test, dgp: "informative_sampling_heterogeneous_effect", interval: method },
        REPORTED_LEVEL,
    );
    let mut diagnostics = Vec::new();
    for result in
        map_replicates(n_sim(), |rep| fit(&sample(n, 91_000 + rep, Effect::Heterogeneous, false)))
    {
        let Some(result) = result else {
            tally.skip();
            continue;
        };
        tally.bind(
            &basis,
            ScopeFacts {
                row_count: n as u64,
                replicates_ok: None,
                posterior_draws: None,
                unidentified_mass: 0.0,
            },
        );
        tally.record(Some((result.lower, result.upper)), TRUTH);
        diagnostics.push((
            result.estimate.ate,
            result.noise_variance,
            result.target_variance,
            result.point_correction,
        ));
    }
    if !diagnostics.is_empty() {
        let count = diagnostics.len() as f64;
        let mean = diagnostics.iter().map(|d| d.0).sum::<f64>() / count;
        let variance =
            diagnostics.iter().map(|d| (d.0 - mean).powi(2)).sum::<f64>() / (count - 1.0);
        eprintln!(
            "info population AIPW {test}: bias={:.6} sd={:.6} noise_variance={:.6} target_variance={:.6} point_correction={:.6}",
            mean - TRUTH,
            variance.sqrt(),
            diagnostics.iter().map(|d| d.1).sum::<f64>() / count,
            diagnostics.iter().map(|d| d.2).sum::<f64>() / count,
            diagnostics.iter().map(|d| d.3).sum::<f64>() / count
        );
    }
    tally.assert();
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
fn weighted_aipw_population_linear_interval() {
    measure_population_aipw("weighted_aipw_population_linear_interval", "aipw", |sample| {
        let (data, estimand) = table(sample)?;
        let estimator = AipwAte::new()
            .with_bootstrap_replicates(0)
            .with_sampling_weights(SamplingWeights::new(sample.w.clone()).ok()?);
        let p = estimator.prepare(&data, &estimand, &query()).ok()?;
        estimator
            .population_linear_interval(
                &p,
                &mut AipwWorkspace::default(),
                &ExecutionContext::for_tests(1),
                AssumptionSet::new(),
                REPORTED_LEVEL,
            )
            .ok()
    });
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn weighted_stratification_interval() {
    measure(
        "weighted_stratification_interval",
        "informative_sampling_heterogeneous_effect",
        1200,
        &construction_for(
            "propensity.stratification",
            "",
            "iid",
            "AverageEffect",
            "all_observed.mean+stratification=regression_linear_hc3",
        ),
        |n, seed| {
            let s = sample(n, seed, Effect::Heterogeneous, false);
            scored(stratification(&s), n)
        },
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn weighted_matching_interval() {
    measure(
        "weighted_matching_interval",
        "informative_sampling_heterogeneous_effect",
        1200,
        &construction_for(
            "propensity.matching",
            "homoskedastic",
            "iid",
            "AverageEffect",
            "all_observed.mean+matching=population_linear",
        ),
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
        panel,
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
        &construction_for(
            "distance.matching",
            "homoskedastic",
            "iid",
            "AverageEffect",
            "all_observed.mean+matching=population_linear",
        ),
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
fn weighted_dml_aipw_population_linear_interval() {
    measure_population_aipw("weighted_dml_aipw_population_linear_interval", "dml", |sample| {
        let (data, estimand) = table(sample)?;
        let estimator =
            DmlAte::new().with_sampling_weights(SamplingWeights::new(sample.w.clone()).ok()?);
        let p = estimator.prepare(&data, &estimand, &query()).ok()?;
        estimator
            .population_linear_interval(
                &p,
                &ExecutionContext::for_tests(1),
                AssumptionSet::new(),
                REPORTED_LEVEL,
            )
            .ok()
    });
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

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn weighted_dr_learner_population_linear_interval() {
    measure_population_aipw(
        "weighted_dr_learner_population_linear_interval",
        "dr.learner",
        |sample| {
            let (data, estimand) = table(sample)?;
            let estimator = DrLearner::new()
                .with_sampling_weights(SamplingWeights::new(sample.w.clone()).ok()?);
            let p = estimator.prepare(&data, &estimand, &query()).ok()?;
            estimator
                .population_linear_interval(
                    &p,
                    &ExecutionContext::for_tests(1),
                    AssumptionSet::new(),
                    REPORTED_LEVEL,
                )
                .ok()
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
    let n = grid_n(1200);
    let method = "population_linear_atc_hc3";
    let mut basis = construction_for(
        "aipw",
        "population_linear_heteroskedastic_hc3",
        "iid",
        "AverageEffect",
        "untreated.mean+overlap=clip:none,trim:none+interval_variance=heteroskedastic_arm_linear_training_hc3",
    );
    basis.interval_method = method.into();
    let mut tally = CoverageTally::for_record(
        RecordKey {
            test: "weighted_aipw_atc_interval",
            dgp: "informative_sampling_constant_effect",
            interval: method,
        },
        REPORTED_LEVEL,
    );
    for result in map_replicates(n_sim(), |rep| {
        let s = sample(n, 91_000 + rep, Effect::Constant, false);
        let (data, estimand) = table(&s)?;
        let estimator = AipwAte::new()
            .with_bootstrap_replicates(0)
            .with_overlap(no_clip())
            .with_sampling_weights(SamplingWeights::new(s.w).ok()?);
        let p = estimator
            .prepare(&data, &estimand, &query().with_target_population(TargetPopulation::Untreated))
            .ok()?;
        estimator
            .population_linear_atc_interval(
                &p,
                &mut AipwWorkspace::default(),
                &ExecutionContext::for_tests(1),
                AssumptionSet::new(),
                REPORTED_LEVEL,
            )
            .ok()
    }) {
        let Some(result) = result else {
            tally.skip();
            continue;
        };
        tally.bind(
            &basis,
            ScopeFacts {
                row_count: n as u64,
                replicates_ok: None,
                posterior_draws: None,
                unidentified_mass: 0.0,
            },
        );
        tally.record(Some((result.lower, result.upper)), TRUTH);
    }
    tally.assert();
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
        &cluster_reference_construction("aipw"),
        |n, seed| {
            let s = sample(n, seed, Effect::Heterogeneous, true);
            scored_cluster_reference(aipw_cluster_dml(&s), n)
        },
    );
}

/// Sampled units with a negative-binomial count outcome: `Y | T, z` has mean
/// `exp(-0.3 + 0.8 T + 0.6 z)` times a mean-one Gamma(2) multiplier, with the informative
/// sampling and assignment of [`sample`].
fn count_sample(n: usize, seed: u64) -> Sample {
    let mut rng = ExecutionContext::for_tests(validation_seed(seed))
        .rng
        .stream_for(StreamDomain::Test, 0x5A05);
    let mut out = Sample { t: vec![], y: vec![], z: vec![], w: vec![], cluster: vec![] };
    while out.t.len() < n {
        let z = standard_normal(&mut rng);
        let t = f64::from(rng.next_f64() < sigmoid(-0.3 + 0.8 * z));
        let gamma = -(rng.next_f64().max(1e-300)).ln() - (rng.next_f64().max(1e-300)).ln();
        let mean = (-0.3 + 0.8 * t + 0.6 * z).exp() * gamma / 2.0;
        // Knuth's Poisson sampler; the means here are small.
        let limit = (-mean).exp();
        let (mut count, mut product) = (0.0, rng.next_f64());
        while product > limit {
            count += 1.0;
            product *= rng.next_f64();
        }
        let inclusion = sigmoid(-1.4 + z);
        if rng.next_f64() < inclusion {
            out.t.push(t);
            out.y.push(count);
            out.z.push(z);
            out.w.push(1.0 / inclusion);
            out.cluster.push(u32::try_from(out.t.len()).unwrap_or(u32::MAX));
        }
    }
    out
}

/// `E_z[exp(0.5 + 0.6 z) - exp(-0.3 + 0.6 z)] = exp(-0.3 + 0.18) (e^0.8 - 1)` for `z ~ N(0, 1)`.
fn count_population_effect() -> f64 {
    (-0.3_f64 + 0.18).exp() * (0.8_f64.exp() - 1.0)
}

fn penalized_aipw(sample: &Sample, nuisance: PropensityNuisance) -> Option<EffectEstimate> {
    let (data, estimand) = table(sample)?;
    let mut estimator = AipwAte::new()
        .with_bootstrap_replicates(0)
        .with_sampling_weights(SamplingWeights::new(sample.w.clone()).ok()?);
    estimator.propensity = nuisance;
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

fn clustered_flexible(sample: &Sample, dr: bool) -> Option<EffectEstimate> {
    let (data, estimand) = table(sample)?;
    let weights = SamplingWeights::new(sample.w.clone()).ok()?;
    let unit = ClusterDml::new(20).ok()?;
    let context = ExecutionContext::for_tests(1);
    if dr {
        let estimator = DrLearner::new()
            .with_sampling_weights(weights)
            .with_cluster_dml(unit)
            .with_source_cluster_ids(sample.cluster.clone());
        let prepared = estimator.prepare(&data, &estimand, &query()).ok()?;
        estimator.fit(&prepared, &context, AssumptionSet::new()).ok()
    } else {
        let estimator = DmlAte::new()
            .with_sampling_weights(weights)
            .with_cluster_dml(unit)
            .with_source_cluster_ids(sample.cluster.clone());
        let prepared = estimator.prepare(&data, &estimand, &query()).ok()?;
        estimator.fit(&prepared, &context, AssumptionSet::new()).ok()
    }
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn weighted_aipw_ridge_propensity_interval() {
    measure(
        "weighted_aipw_ridge_propensity_interval",
        "informative_sampling_heterogeneous_effect",
        1200,
        &construction("aipw", "homoskedastic", "iid", "AverageEffect"),
        |n, seed| {
            let s = sample(n, seed, Effect::Heterogeneous, false);
            let tuning = RidgeTuning::new(&[0.1, 1.0, 10.0], 3).ok()?;
            scored(penalized_aipw(&s, PropensityNuisance::ridge_logistic(tuning)), n)
        },
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn weighted_aipw_lasso_propensity_interval() {
    measure(
        "weighted_aipw_lasso_propensity_interval",
        "informative_sampling_heterogeneous_effect",
        1200,
        &construction("aipw", "homoskedastic", "iid", "AverageEffect"),
        |n, seed| {
            let s = sample(n, seed, Effect::Heterogeneous, false);
            let tuning = RidgeTuning::new(&[0.1, 1.0, 10.0], 3).ok()?;
            scored(penalized_aipw(&s, PropensityNuisance::lasso_with(tuning)), n)
        },
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn weighted_glm_negative_binomial_interval() {
    measure_against(
        count_population_effect(),
        "weighted_glm_negative_binomial_interval",
        "informative_sampling_overdispersed_counts",
        1200,
        &construction("glm.adjustment", "hc1", "iid", "AverageEffect"),
        |n, seed| {
            let s = count_sample(n, seed);
            scored(glm(&s, GlmFamily::NegativeBinomial), n)
        },
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn weighted_dml_cluster_interval() {
    measure(
        "weighted_dml_cluster_interval",
        "informative_sampling_clustered_heterogeneous_effect",
        1200,
        &cluster_reference_construction("dml"),
        |n, seed| {
            let s = sample(n, seed, Effect::Heterogeneous, true);
            scored_cluster_reference(clustered_flexible(&s, false), n)
        },
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn weighted_dr_learner_cluster_interval() {
    measure(
        "weighted_dr_learner_cluster_interval",
        "informative_sampling_clustered_heterogeneous_effect",
        1200,
        &cluster_reference_construction("dr.learner"),
        |n, seed| {
            let s = sample(n, seed, Effect::Heterogeneous, true);
            scored_cluster_reference(clustered_flexible(&s, true), n)
        },
    );
}

#[test]
fn historical_cluster_protocol_replays_the_raw_normal_interval_and_original_identity() {
    let mut fit = EffectEstimate::new(2.0, 0.5, AssumptionSet::new(), no_clip());
    fit.uncertainty.analytic.reference_df = Some(4.0);
    // The exact constant used by the historical producer, independently of dispatch.
    let historical_critical = 1.959_963_984_540_054_f64;
    let expected = (2.0 - historical_critical * 0.5, 2.0 + historical_critical * 0.5);
    let replay = scored_cluster_reference_for_protocol(Some(fit), 17, 1).unwrap();
    assert_eq!(replay.0.0.to_bits(), expected.0.to_bits());
    assert_eq!(replay.0.1.to_bits(), expected.1.to_bits());
    assert_eq!(replay.1, 17);
    for estimator in ["aipw", "dml", "dr.learner"] {
        let original = construction(estimator, "cluster", "cluster", "AverageEffect");
        let replay = cluster_reference_construction_for_protocol(estimator, 1);
        assert_eq!(replay.functional, original.functional);
        assert_eq!(replay.interval_method, original.interval_method);
        assert_eq!(replay.estimator, original.estimator);
        assert_eq!(replay.se_kind, original.se_kind);
        let current = cluster_reference_construction_for_protocol(estimator, 2);
        assert_ne!(replay.functional, current.functional);
        assert_ne!(replay.interval_method, current.interval_method);
    }
}
