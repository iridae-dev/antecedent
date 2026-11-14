//! Coverage of further credible intervals of the Bayesian randomized-design routes: the
//! finite-population contrast with Bernoulli arms (sharp null held two-sided, varying unit
//! effects held to at least nominal), and the cluster Bayesian bootstrap at twenty-five to
//! forty-nine clusters per arm.
//!
//! Builders mirror the family's first harness file, parametrized where a statement needs it.
//! Records are keyed on the design axes (structure `design`, validation `not_applicable`). A
//! replicate that publishes nothing to score is a skip (a miss).
//!
//! Ignored tests run via `scripts/measure_calibration.sh`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    reason = "simulation dimensions are small counts"
)]

mod common;

use antecedent::{
    BayesianConfig, DesignPosteriorModel, EstimatorId, InferenceMode, InterferenceSpec,
    RandomizedEffectQuery, RefuteSuite, Study, StudyBuilder, StudyResult,
};
use antecedent_core::{
    AssignmentDesign, CausalQuery, ExecutionContext, ExposureLevel, ExposureMapping,
    InterferenceFunctional, InterferenceQuery, RandomizationDesign, VariableId,
};
use antecedent_data::{NetworkData, NetworkEdge, TabularData};
use antecedent_estimate::randomized_posterior::{ArmBetaPrior, ArmOutcomeModel};
use antecedent_graph::Dag;
use common::calibration::{
    Construction, CoverageSide, CoverageTally, REPORTED_LEVEL, RecordKey, SampleGrid, ScopeFacts,
    gaussian, grid_n, grid_seed, map_replicates, n_sim, stream_seed, unit_uniform,
};
use common::calibration_bind::{bind_reported, constructions};
use std::sync::Arc;

const INTERVAL: &str = "posterior_quantile";

const DRAWS: usize = 2_000;

type Reported = Vec<(Construction, ScopeFacts, &'static str)>;

/// One scored replicate: the published interval, the known truth and the runtime's construction.
struct Replicate {
    interval: (f64, f64),
    truth: f64,
    reported: Reported,
}

fn v(raw: u32) -> VariableId {
    VariableId::from_raw(raw)
}

fn labels(n: usize, prefix: &str) -> Vec<Arc<str>> {
    (0..n).map(|i| Arc::<str>::from(format!("{prefix}-{i}"))).collect()
}

/// Prepare and estimate one study, returning its result and the construction the runtime reports.
/// A refusal is `None`.
fn execute(
    builder: StudyBuilder,
    data: &TabularData,
    seed: u64,
) -> Option<(StudyResult, Reported)> {
    let ctx = ExecutionContext::for_tests(seed);
    let prepared = builder.build().ok()?.prepare(&ctx).ok()?;
    let result = prepared.estimate(data, &ctx).ok()?;
    let contract = prepared.contract_for_result(&result).expect("the execution contract");
    let reported = constructions(&contract, &result);
    Some((result, reported))
}

/// The posterior's published 95% interval of one column, `None` when it is absent or not ordered.
fn posterior_interval(result: &StudyResult, column: usize) -> Option<(f64, f64)> {
    let posterior = result.posterior.as_ref()?;
    let (lower, upper) = (posterior.summaries.q025[column], posterior.summaries.q975[column]);
    (lower.is_finite() && upper.is_finite() && lower <= upper).then_some((lower, upper))
}

/// Score every replicate against its truth under one record.
///
/// A graphless design route has no graph and runs no validation suite, and the runtime keys
/// it with no query. Its record is keyed on the design axes instead: `design_query` names the
/// query, the structure is `design` and no validation mode applies. Every other field of the
/// construction is the runtime's. A route on a graph (`None`) binds the runtime's key as is.
fn measure(
    test: &'static str,
    dgp: &'static str,
    design_query: Option<&'static str>,
    runs: &[Option<Replicate>],
) {
    tally_of(test, dgp, design_query, runs).assert();
}

/// The record tally of [`measure`], before its assertion.
fn tally_of(
    test: &'static str,
    dgp: &'static str,
    design_query: Option<&'static str>,
    runs: &[Option<Replicate>],
) -> CoverageTally {
    let mut tally =
        CoverageTally::for_record(RecordKey { test, dgp, interval: INTERVAL }, REPORTED_LEVEL);
    for run in runs {
        let Some(replicate) = run else {
            tally.skip();
            continue;
        };
        match design_query {
            Some(query) => {
                let (construction, scope, _) = replicate
                    .reported
                    .iter()
                    .find(|(construction, _, _)| construction.interval_method == INTERVAL)
                    .expect("the execution reports its posterior interval");
                let mut construction = construction.clone();
                construction.query = query.into();
                construction.structure = "design".into();
                tally.bind_validation(&construction, *scope, "not_applicable");
            }
            None => bind_reported(&mut tally, &replicate.reported),
        }
        tally.record(Some(replicate.interval), replicate.truth);
    }
    tally
}

/// `treated` units chosen uniformly without replacement.
fn complete_assignment(n: usize, treated: usize, seed: u64) -> Vec<bool> {
    let keys: Vec<f64> = (0..n).map(|i| unit_uniform(stream_seed(seed, i as u64))).collect();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| keys[a].total_cmp(&keys[b]));
    let mut assigned = vec![false; n];
    for &unit in &order[..treated] {
        assigned[unit] = true;
    }
    assigned
}
// ------------------------------------------------------------------ randomized arm contrasts

fn complete_query(assigned: Vec<bool>) -> RandomizedEffectQuery {
    let n = assigned.len();
    let treated = assigned.iter().filter(|a| **a).count();
    RandomizedEffectQuery::with_design(
        RandomizationDesign::Complete { treated_units: treated },
        v(0),
        assigned,
        vec![treated as f64 / n as f64; n],
        labels(n, "unit"),
        labels(n, "row"),
        ("control", "treated"),
    )
}

fn arm_inference(model: ArmOutcomeModel) -> InferenceMode {
    InferenceMode::Bayesian(
        BayesianConfig::conjugate()
            .n_draws(DRAWS)
            .design_model(DesignPosteriorModel::ArmOutcome { model }),
    )
}

fn arm_replicate(
    outcome: &[f64],
    assigned: Vec<bool>,
    model: ArmOutcomeModel,
    estimator: Option<EstimatorId>,
    truth: f64,
    seed: u64,
) -> Option<Replicate> {
    let data = TabularData::from_f64_columns([("outcome", outcome)]).ok()?;
    let mut builder = Study::tabular(data.clone())
        .query(complete_query(assigned))
        .inference(arm_inference(model));
    if let Some(estimator) = estimator {
        builder = builder.estimator(estimator);
    }
    let (result, reported) = execute(builder, &data, seed)?;
    Some(Replicate { interval: posterior_interval(&result, 0)?, truth, reported })
}

/// Fixed binary units: two in five succeed under either arm, and one in five more succeed only
/// when treated if `responders`. The truth is the mean unit effect over those units: zero under
/// the sharp null, otherwise the share of responders (one fifth when five divides `n`).
fn fixed_binary_outcomes(n: usize, responders: bool) -> (Vec<f64>, Vec<f64>, f64) {
    let control: Vec<f64> = (0..n).map(|i| f64::from(u8::from(i % 5 < 2))).collect();
    let treated: Vec<f64> =
        (0..n).map(|i| f64::from(u8::from(i % 5 < 2 || (responders && i % 5 == 2)))).collect();
    let truth = treated.iter().zip(&control).map(|(t, c)| t - c).sum::<f64>() / n as f64;
    (control, treated, truth)
}

fn finite_population_bernoulli_runs(responders: bool, seed_base: u64) -> Vec<Option<Replicate>> {
    let n = grid_n(300);
    let (control, treated, truth) = fixed_binary_outcomes(n, responders);
    map_replicates(n_sim(), |rep| {
        let seed = grid_seed(seed_base + rep);
        let assigned = complete_assignment(n, n / 2, seed);
        let outcome: Vec<f64> =
            (0..n).map(|i| if assigned[i] { treated[i] } else { control[i] }).collect();
        arm_replicate(
            &outcome,
            assigned,
            ArmOutcomeModel::Bernoulli(ArmBetaPrior { alpha: 1.0, beta: 1.0 }),
            None,
            truth,
            seed,
        )
    })
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn finite_population_bernoulli_arms_sharp_null_credible_interval() {
    // Every unit effect is zero, so effects are additive, Neyman's bound is the randomization
    // variance and the truth is exactly zero.
    let runs = finite_population_bernoulli_runs(false, 0xB1AC_0000);
    measure(
        "finite_population_bernoulli_arms_sharp_null_credible_interval",
        "complete_randomization_fixed_binary_units_sharp_null",
        Some("RandomizedEffect"),
        &runs,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn finite_population_bernoulli_arms_varying_effects_credible_interval() {
    // Conservative by construction: unit effects are zero or one, so their variance (0.16) is
    // subtracted from Neyman's bound (0.95 over n) in the randomization variance of the contrast.
    let runs = finite_population_bernoulli_runs(true, 0xB1AD_0000);
    tally_of(
        "finite_population_bernoulli_arms_varying_effects_credible_interval",
        "complete_randomization_fixed_binary_units_known_finite_population_risk_difference",
        Some("RandomizedEffect"),
        &runs,
    )
    .assert_one_sided(
        CoverageSide::Upper,
        "the finite-population risk difference covered at least nominally when unit effects vary",
    );
}
// ------------------------------------------------------------------------------- switchback

// ------------------------------------------------------------------ cluster Bayesian bootstrap

const CLUSTER_TOTAL_EFFECT: f64 = 3.0;

fn cluster_replicate(clusters: usize, seed: u64) -> Option<Replicate> {
    let treated = complete_assignment(clusters, clusters / 2, stream_seed(seed, 0xC1));
    let mut noise = gaussian(seed);
    let (mut outcome, mut assignment) = (Vec::new(), Vec::new());
    for &arm in &treated {
        let shared = noise();
        for _ in 0..2 {
            outcome.push(shared + 0.5 * noise() + CLUSTER_TOTAL_EFFECT * f64::from(u8::from(arm)));
            assignment.push(arm);
        }
    }
    let units = 2 * clusters;
    let edges: Vec<NetworkEdge> = (0..clusters as u64)
        .flat_map(|c| {
            [NetworkEdge::new(2 * c, 2 * c + 1, 1.0), NetworkEdge::new(2 * c + 1, 2 * c, 1.0)]
        })
        .collect();
    let data = TabularData::from_f64_columns([("y", outcome.as_slice())]).ok()?;
    let query = InterferenceQuery::new(
        AssignmentDesign::ClusterRandomization {
            clusters: Arc::from((0..units).map(|u| (u / 2) as u32).collect::<Vec<_>>()),
            treated_clusters: clusters / 2,
        },
        ExposureMapping::NeighborFraction,
        InterferenceFunctional::ExposureContrast {
            outcome: v(0),
            from: ExposureLevel { own: 0.0, neighbors: 0.0 },
            to: ExposureLevel { own: 1.0, neighbors: 1.0 },
        },
    );
    let network = NetworkData::try_new(data.clone(), edges).ok()?;
    let builder = Study::tabular(data.clone())
        .graph(Dag::with_variables(1))
        .query(CausalQuery::Interference(query))
        .interference(InterferenceSpec { network, assignment: Arc::from(assignment) })
        .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(DRAWS)))
        .refute(RefuteSuite::None);
    let (result, reported) = execute(builder, &data, seed)?;
    let column = result.posterior.as_ref()?.effect_column()?;
    Some(Replicate {
        interval: posterior_interval(&result, column)?,
        truth: CLUSTER_TOTAL_EFFECT,
        reported,
    })
}

/// Twenty-five, thirty-seven and forty-nine clusters per arm: the upper part of the range in
/// which the route publishes its Student-t reference law.
const LARGER_SMALL_ARMS: SampleGrid =
    SampleGrid { name: "cluster_arms_25_to_49", factors: [(25, 49), (37, 49), (1, 1)] };

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn cluster_bayesian_bootstrap_larger_small_arm_credible_interval() {
    // The truth is the constant total effect; both arms hold the same number of clusters.
    let clusters = 2 * LARGER_SMALL_ARMS.n(49);
    let runs =
        map_replicates(n_sim(), |rep| cluster_replicate(clusters, grid_seed(0xB1AE_0000 + rep)));
    measure(
        "cluster_bayesian_bootstrap_larger_small_arm_credible_interval",
        "paired_cluster_randomization_constant_total_effect_25_to_49_clusters_per_arm",
        None,
        &runs,
    );
}
// --------------------------------------------------------------------- model-based arm effect
