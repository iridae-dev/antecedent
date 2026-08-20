//! Interactive envelope prior must anchor to the first identified atom in
//! original order — before stratified subsample can drop that atom (0.6.0).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::too_many_lines
)]

use std::sync::Arc;

use antecedent::discovery::GraphPosterior;
use antecedent::{BayesianConfig, InferenceMode, LatencyMode, RefuteSuite, Study};
use antecedent_core::{
    AverageEffectQuery, CausalSchemaBuilder, ExecutionContext, MeasurementSpec, RoleHint,
    SmallRoleSet, ValueType, VariableId,
};
use antecedent_data::{
    Float64Column, OwnedColumn, OwnedColumnarStorage, TabularData, ValidityBitmap,
};
use antecedent_discovery::set_edge;
use antecedent_estimate::{BayesianBackendKind, BayesianGComputationAte};
use antecedent_expr::{ExprId, IdentifiedEstimand};
use antecedent_prob::{
    ExternalPriorSource, ExternalPriorWeight, GaussianCoefficientPrior, GraphIdentFlag, PriorSet,
    PriorSpec, WeightedGraphSamples, compose_external_priors,
};
use antecedent_validate::ConflictPolicy;

const N_ID: usize = 17; // > INTERACTIVE_MAX_ENVELOPE_GRAPHS (16)
const SUBSAMPLE_STREAM: u64 = 0xE11E;

fn toy_table(n: usize) -> TabularData {
    let mut b = CausalSchemaBuilder::new();
    b.add_variable(
        "T",
        ValueType::Continuous,
        SmallRoleSet::from_hint(RoleHint::TreatmentCandidate),
        None,
        None,
        MeasurementSpec::default(),
    )
    .unwrap();
    b.add_variable(
        "Y",
        ValueType::Continuous,
        SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
        None,
        None,
        MeasurementSpec::default(),
    )
    .unwrap();
    b.add_variable(
        "Z",
        ValueType::Continuous,
        SmallRoleSet::from_hint(RoleHint::Context),
        None,
        None,
        MeasurementSpec::default(),
    )
    .unwrap();
    b.add_variable(
        "W",
        ValueType::Continuous,
        SmallRoleSet::from_hint(RoleHint::Context),
        None,
        None,
        MeasurementSpec::default(),
    )
    .unwrap();
    let schema = b.build().unwrap();
    let t = VariableId::from_raw(0);
    let y = VariableId::from_raw(1);
    let z = VariableId::from_raw(2);
    let w = VariableId::from_raw(3);
    // Z confounds; W is nearly noise — conflict PPC differs on Z vs W designs.
    let tv: Vec<f64> = (0..n).map(|i| (i % 2) as f64).collect();
    let zv: Vec<f64> = (0..n).map(|i| (i as f64) * 0.08 - 2.0).collect();
    let wv: Vec<f64> = (0..n).map(|i| ((i * 7) % 5) as f64 * 0.01).collect();
    let yv: Vec<f64> = (0..n).map(|i| 1.0 + 2.0 * tv[i] + 1.2 * zv[i] + 0.02 * wv[i]).collect();
    let validity = ValidityBitmap::all_valid(n);
    let cols = vec![
        OwnedColumn::Float64(Float64Column::new(t, Arc::from(tv), validity.clone()).unwrap()),
        OwnedColumn::Float64(Float64Column::new(y, Arc::from(yv), validity.clone()).unwrap()),
        OwnedColumn::Float64(Float64Column::new(z, Arc::from(zv), validity.clone()).unwrap()),
        OwnedColumn::Float64(Float64Column::new(w, Arc::from(wv), validity).unwrap()),
    ];
    let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
    TabularData::new(storage)
}

fn z_confounder_mask() -> u64 {
    // T=0, Y=1, Z=2: Z→T, Z→Y, T→Y
    let mut m = 0u64;
    m = set_edge(m, 4, 2, 0, true);
    m = set_edge(m, 4, 2, 1, true);
    m = set_edge(m, 4, 0, 1, true);
    m
}

fn w_confounder_mask() -> u64 {
    // T=0, Y=1, W=3: W→T, W→Y, T→Y
    let mut m = 0u64;
    m = set_edge(m, 4, 3, 0, true);
    m = set_edge(m, 4, 3, 1, true);
    m = set_edge(m, 4, 0, 1, true);
    m
}

fn graph_posterior(masks: Vec<u64>) -> GraphPosterior {
    let n = masks.len();
    let w = 1.0 / n as f64;
    GraphPosterior::new(
        4,
        vec![w; n],
        masks,
        vec![0.0; 16],
        vec![0.0; 16],
        n as f64,
        antecedent_prob::InferenceDiagnostics::analytic("test"),
        0,
    )
    .unwrap()
}

fn first_identified_dropped(seed: u64, keys: &[u64]) -> bool {
    let weights = vec![1.0 / keys.len() as f64; keys.len()];
    let flags = vec![GraphIdentFlag::Identified; keys.len()];
    let graphs = WeightedGraphSamples::new(weights, flags, keys.to_vec()).unwrap();
    let ctx = ExecutionContext::for_tests(seed);
    let mut rng = ctx.rng.stream(SUBSAMPLE_STREAM);
    let sub = graphs.stratified_interactive_subsample(16, &mut rng).unwrap();
    assert!(sub.approximate);
    !sub.graphs
        .identified
        .iter()
        .zip(keys.iter())
        .any(|(flag, key)| *flag == GraphIdentFlag::Identified && *key == keys[0])
}

fn composed_cfg(data: &TabularData) -> BayesianConfig {
    let t = VariableId::from_raw(0);
    let y = VariableId::from_raw(1);
    let z = VariableId::from_raw(2);
    let probe = BayesianGComputationAte {
        backend: BayesianBackendKind::ConjugateGaussian,
        n_draws: 48,
        seed: 3,
        ..BayesianGComputationAte::new()
    };
    let estimand =
        IdentifiedEstimand::backdoor("backdoor.adjustment", Arc::from([z]), ExprId::from_raw(0));
    let query = AverageEffectQuery::binary_ate(t, y);
    let prep = probe.prepare(data, &estimand, &query).unwrap();
    let ncols = prep.design.ncols;
    let t_col = prep.design.treatment_column().expect("treatment column");
    // Far-from-truth treatment mean so conflict shrink is prep-sensitive.
    let mut mean = vec![0.0; ncols];
    mean[t_col] = 8.0;
    let mut source_prior = PriorSet::new();
    source_prior.push(PriorSpec::GaussianCoefficients(GaussianCoefficientPrior {
        mean: Arc::from(mean),
        variance: Arc::from(vec![0.05; ncols]),
    }));
    let sources = Arc::<[ExternalPriorSource]>::from(vec![ExternalPriorSource {
        id: Arc::from("anchor_bank"),
        prior: source_prior,
        weight: ExternalPriorWeight::power(1.0).unwrap(),
        ess: None,
    }]);
    let baseline = PriorSet::weakly_informative(ncols);
    let composed = compose_external_priors(&sources, &baseline).unwrap();
    let policy = ConflictPolicy::try_new(0.05, 1.0).unwrap();
    BayesianConfig::conjugate().n_draws(48).prior_from_composed(sources, composed, Some(policy))
}

fn run_envelope(
    data: TabularData,
    gp: GraphPosterior,
    cfg: BayesianConfig,
    seed: u64,
) -> (f64, bool) {
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let result = Study::tabular(data)
        .graph_posterior(gp)
        .query(query)
        .inference(InferenceMode::Bayesian(cfg))
        .latency_mode(LatencyMode::Interactive)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(seed))
        .unwrap();
    let subsampled = result
        .diagnostics
        .iter()
        .any(|d| d.code.as_ref() == "estimate.envelope.interactive_subsample");
    let post = result.posterior.expect("posterior");
    let mean = post.summaries.mean[post.effect_column().unwrap()];
    (mean, subsampled)
}

#[test]
fn interactive_envelope_prior_anchors_to_first_identified_before_subsample() {
    let data = toy_table(80);
    let z_mask = z_confounder_mask();
    let w_mask = w_confounder_mask();
    assert_ne!(z_mask, w_mask);

    let mut masks_z_first = vec![z_mask];
    masks_z_first.extend(std::iter::repeat_n(w_mask, N_ID - 1));
    let keys: Vec<u64> = masks_z_first.clone();

    let seed = (0u64..10_000)
        .find(|&s| first_identified_dropped(s, &keys))
        .expect("need a seed that drops the first identified atom under Interactive subsample");

    let cfg = composed_cfg(&data);
    let (mean_z_first, subsampled) =
        run_envelope(data.clone(), graph_posterior(masks_z_first), cfg.clone(), seed);
    assert!(subsampled, "Interactive subsample diagnostic must fire with {N_ID} identified graphs");

    // Same seed, but first-in-order atom is a W-graph. Fixed code anchors the
    // shared prior to that first atom *before* subsample; buggy post-subsample
    // anchoring would often pick the same kept W-atom for both orderings.
    let mut masks_w_first = vec![w_mask];
    masks_w_first.extend(std::iter::repeat_n(w_mask, N_ID - 2));
    masks_w_first.push(z_mask);
    let (mean_w_first, _) = run_envelope(data, graph_posterior(masks_w_first), cfg, seed);

    assert!(
        (mean_z_first - mean_w_first).abs() > 1e-6,
        "prior must follow first-in-order identified atom (z-first={mean_z_first}, w-first={mean_w_first}); \
         equal means suggest post-subsample prior anchoring"
    );
    assert!(mean_z_first.is_finite() && mean_w_first.is_finite());
}
