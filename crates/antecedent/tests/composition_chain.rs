//! The whole composition chain on one planted dataset, through the public facade.
//!
//! Messy evidence -> uncertain measurements -> plausible causal structures -> evidence-weighted
//! estimation -> population-specific conclusions:
//!
//! 1. the exposure `x` is observed only through a misclassified coding, with the error rates
//!    estimated from a validation subsample; `StudyBuilder::measured_exposure` imputes exposure
//!    draws and hands back one completed table per draw;
//! 2. a posterior structure search over `(z, x, y)` with a declared structure prior (per-edge
//!    beliefs and a soft tier order `z` before `x` before `y`) runs on those completed tables and
//!    is pooled across them in the engine;
//! 3. the effect is estimated by cross-fitted AIPW whose outcome nuisance is regularised by a
//!    named composed external prior, in every structure the pooled posterior supports, and the
//!    structure-wise effects are mixed by the pooled posterior's weights;
//! 4. the estimate is for a declared kernel-weighted target population over `z`, whose truth is
//!    the closed-form kernel-weighted mean of the planted modified effect on the planted rows.
//!
//! The planted effect is `tau(z) = 1.5 + 0.5 z`. The stages hand each other only the objects the
//! library produced. The one place the library has no first-class hand-off is stage 3: a
//! population-targeted estimate over a pooled structure posterior, per completed table. A graph
//! posterior study refuses a target population other than all rows, and the measured-exposure
//! pooling takes one study result per table, so this test composes it from public pieces
//! (`GraphPosterior::atoms` / `weights`, one `Study` per supported structure, the
//! structure-weighted mixture with its between-structure variance, `pool_measured_effect`).
//!
//! The naive pipeline (the recorded column, one assumed structure, no prior, every row) misses the
//! truth by more than the tolerance, so each stage earns its place.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::too_many_lines,
    reason = "integration test with a planted truth and a caller-side composition"
)]

mod common;

use std::sync::Arc;

use antecedent::discovery::{
    AdjacencyMask, BayesianDiscoverParams, EdgePriors, GraphPosterior, GraphPrior, SoftTiers,
};
use antecedent::estimate::measurement::{
    ExposureErrorDeclaration, ExposureMeasurement, MeasuredEffect, RatePair, pool_measured_effect,
};
use antecedent::inference::{KernelFamily, TransportKernel, outcome_prior_source_from_composed};
use antecedent::measured_exposure::{MeasuredColumns, MeasuredExposureRequest};
use antecedent::{
    AverageEffectQuery, CausalError, DiscoverySpec, RefuteSuite, Study, StudyBuilder,
    WeightedTargetDeclaration,
};
use antecedent_core::{Assumption, ExecutionContext, VariableId};
use antecedent_data::TabularData;
use antecedent_estimate::{AipwAte, OutcomePriorSource};
use antecedent_graph::{Dag, DenseNodeId};
use antecedent_prob::{
    ComposedPrior, ExternalPriorSource, ExternalPriorWeight, GaussianCoefficientPrior, PriorSet,
    PriorSpec, compose_external_priors,
};
use common::calibration::{gaussian, uniform};

const ROWS: usize = 4_000;
const VALIDATION_ROWS: usize = 800;
const DRAWS: usize = 8;
const SENSITIVITY: f64 = 0.85;
const SPECIFICITY: f64 = 0.9;
/// The tolerance on the population effect, in effect units.
const TOLERANCE: f64 = 0.35;
const CENTRE: f64 = 1.0;
const BANDWIDTH: f64 = 0.6;
/// A structure with less posterior mass than this is not estimated; its mass is reported.
const NEGLIGIBLE_MASS: f64 = 1e-4;

const Z: usize = 0;
const X: usize = 1;
const Y: usize = 2;

fn v(raw: usize) -> VariableId {
    VariableId::from_raw(u32::try_from(raw).unwrap())
}

fn three() -> Vec<VariableId> {
    vec![v(Z), v(X), v(Y)]
}

fn ctx(seed: u64) -> ExecutionContext {
    ExecutionContext::for_tests(seed)
}

fn on_large_stack(body: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .name("composition-chain".into())
        .stack_size(64 * 1024 * 1024)
        .spawn(body)
        .unwrap()
        .join()
        .unwrap();
}

fn expit(value: f64) -> f64 {
    1.0 / (1.0 + (-value).exp())
}

/// The planted modified effect.
fn tau(z: f64) -> f64 {
    1.5 + 0.5 * z
}

/// The planted evidence: the covariate, the exposure recorded with error, the outcome, the
/// validation subsample's confusion counts, and the planted rows' kernel-weighted truth.
struct Planted {
    z: Vec<f64>,
    recorded: Vec<bool>,
    y: Vec<f64>,
    /// `(true positive, false negative, true negative, false positive)` of the validation sample.
    validation: (u64, u64, u64, u64),
}

fn plant(seed: u64) -> Planted {
    let mut normal = gaussian(seed);
    let mut next = uniform(seed, 1);
    let z: Vec<f64> = (0..ROWS).map(|_| normal()).collect();
    let x: Vec<bool> = z.iter().map(|&zi| next() < expit(0.6 * zi)).collect();
    let y: Vec<f64> = (0..ROWS)
        .map(|i| 1.0 + 0.5 * z[i] + f64::from(u8::from(x[i])) * tau(z[i]) + 0.5 * normal())
        .collect();
    let recorded: Vec<bool> =
        x.iter().map(|&xi| if xi { next() < SENSITIVITY } else { next() >= SPECIFICITY }).collect();
    // A separate validation subsample, where the exposure is known as well as recorded.
    let (mut tp, mut fnn, mut tn, mut fp) = (0_u64, 0_u64, 0_u64, 0_u64);
    for i in 0..VALIDATION_ROWS {
        let truly = i % 2 == 0;
        let coded = if truly { next() < SENSITIVITY } else { next() >= SPECIFICITY };
        match (truly, coded) {
            (true, true) => tp += 1,
            (true, false) => fnn += 1,
            (false, false) => tn += 1,
            (false, true) => fp += 1,
        }
    }
    Planted { z, recorded, y, validation: (tp, fnn, tn, fp) }
}

fn kernel() -> TransportKernel {
    TransportKernel::new(KernelFamily::Gaussian, BANDWIDTH).unwrap()
}

/// The declared target population: a kernel over `z` around [`CENTRE`].
fn target() -> WeightedTargetDeclaration {
    WeightedTargetDeclaration::kernel(kernel(), vec![v(Z)], vec![CENTRE], None).unwrap()
}

/// The closed-form population effect: the kernel-weighted mean of the planted `tau` on the rows,
/// with the weights computed here from the rows' distances to the centre.
fn closed_form_truth(planted: &Planted) -> f64 {
    let distances: Vec<f64> = planted.z.iter().map(|z| (z - CENTRE).abs()).collect();
    let weights = kernel().weights(&distances).unwrap();
    let mass: f64 = weights.iter().sum();
    weights.iter().zip(&planted.z).map(|(w, z)| w * tau(*z)).sum::<f64>() / mass
}

fn flag(value: bool) -> f64 {
    f64::from(u8::from(value))
}

// ---- stage 1: messy evidence -> uncertain measurements ------------------------------------

fn declaration(planted: &Planted) -> ExposureMeasurement {
    let (tp, fnn, tn, fp) = planted.validation;
    ExposureMeasurement::Misclassified {
        recorded: planted.recorded.clone(),
        error: ExposureErrorDeclaration::Shared(RatePair::validated(tp, fnn, tn, fp).unwrap()),
        min_youden: 0.1,
    }
}

fn columns(planted: &Planted) -> MeasuredColumns {
    MeasuredColumns::new(
        vec!["z".into(), "x".into(), "y".into()],
        vec![
            planted.z.clone(),
            planted.recorded.iter().copied().map(flag).collect(),
            planted.y.clone(),
        ],
    )
    .unwrap()
}

// ---- stage 2: measurements -> plausible structures ----------------------------------------

/// Per-edge beliefs (against the reversed, the into-`z` edges) and a soft order `z`, `x`, `y`.
fn structure_params() -> BayesianDiscoverParams {
    let tiers: Vec<Arc<[VariableId]>> =
        vec![Arc::from(vec![v(Z)]), Arc::from(vec![v(X)]), Arc::from(vec![v(Y)])];
    let prior = GraphPrior::uniform()
        .with_edge_priors(EdgePriors::new(vec![
            (v(X), v(Y), 0.5),
            (v(Y), v(X), 0.02),
            (v(X), v(Z), 0.02),
            (v(Y), v(Z), 0.02),
        ]))
        .with_soft_tiers(SoftTiers::with_log_penalty(tiers, 2.0));
    BayesianDiscoverParams { prior, ..BayesianDiscoverParams::default() }
}

/// The prior the compiled search query states.
fn structure_prior_statement(params: &BayesianDiscoverParams) -> String {
    DiscoverySpec::exact_dag_posterior(params.clone())
        .query(three())
        .unwrap()
        .method
        .structure_prior()
        .expect("the query carries the declared prior")
        .statement()
}

fn mass_of_edge(posterior: &GraphPosterior, from: usize, to: usize) -> f64 {
    posterior
        .atoms()
        .iter()
        .zip(posterior.weights().iter())
        .filter(|(atom, _)| atom.has_edge(3, from, to))
        .map(|(_, weight)| *weight)
        .sum()
}

// ---- stage 3: structures -> evidence-weighted estimation ----------------------------------

fn names(items: &[&str]) -> Vec<Arc<str>> {
    items.iter().map(|name| Arc::from(*name)).collect()
}

fn gaussian_prior(mean: Vec<f64>, variance: Vec<f64>) -> PriorSet {
    let mut prior = PriorSet::new();
    prior.push(PriorSpec::GaussianCoefficients(GaussianCoefficientPrior {
        mean: Arc::from(mean),
        variance: Arc::from(variance),
    }));
    prior
}

fn external(id: &str, mean: [f64; 2], variance: [f64; 2], alpha: f64) -> ExternalPriorSource {
    ExternalPriorSource {
        id: Arc::from(id),
        prior: gaussian_prior(mean.to_vec(), variance.to_vec()),
        weight: ExternalPriorWeight::power(alpha).unwrap(),
        ess: None,
    }
}

/// Two external sources on the outcome regression, composed and named by coefficient.
fn composed_prior() -> ComposedPrior {
    let sources = vec![
        external("registry-a", [1.6, 0.7], [0.6, 0.5], 1.0),
        external("trial-b", [1.9, 0.8], [0.8, 0.6], 0.5),
    ];
    compose_external_priors(&sources, &gaussian_prior(vec![0.0; 2], vec![100.0; 2]))
        .unwrap()
        .with_coefficient_names(names(&["intercept", "coef_z"]))
        .unwrap()
}

fn dag_of(atom: &AdjacencyMask) -> Dag {
    let mut dag = Dag::with_variables(3);
    for from in 0..3 {
        for to in 0..3 {
            if from != to && atom.has_edge(3, from, to) {
                dag.insert_directed(
                    DenseNodeId::from_raw(from as u32),
                    DenseNodeId::from_raw(to as u32),
                )
                .expect("an atom of the posterior is acyclic");
            }
        }
    }
    dag
}

fn reaches(atom: &AdjacencyMask, from: usize, to: usize) -> bool {
    let mut seen = [false; 3];
    seen[from] = true;
    let mut stack = vec![from];
    while let Some(node) = stack.pop() {
        for next in 0..3 {
            if !seen[next] && atom.has_edge(3, node, next) {
                seen[next] = true;
                stack.push(next);
            }
        }
    }
    seen[to]
}

/// One structure's effect on one completed table: the AIPW estimate for the declared target
/// population with the composed prior on the outcome nuisance.
fn estimate_structure(
    table: &TabularData,
    atom: &AdjacencyMask,
    prior: &OutcomePriorSource,
    seed: u64,
) -> Result<(f64, f64, antecedent::StudyResult), CausalError> {
    let result = Study::tabular(table.clone())
        .graph(dag_of(atom))
        .estimator(AipwAte::new().with_bootstrap_replicates(0).with_outcome_prior(prior.clone()))
        .refute(RefuteSuite::None)
        .weighted_target(target())
        .query(AverageEffectQuery::binary_ate(v(X), v(Y)))
        .build()?
        .run(&ctx(seed))?;
    let effect = result.estimate().as_effect().expect("an average effect");
    let (ate, se) = (effect.ate, effect.se_analytic());
    Ok((ate, se, result))
}

/// What the composed chain reports: the pooled effect and the objects its provenance names.
struct ChainReport {
    effect: MeasuredEffect,
    /// Posterior mass of the structures that were not estimated (negligible or not answerable
    /// for the declared target).
    unanswered_mass: f64,
    /// The first structure-wise result, for its recorded assumptions.
    sample: antecedent::StudyResult,
}

/// Mix the structure-wise effects of every completed table by the pooled posterior's weights and
/// pool the tables by Rubin's rules.
fn composed_effect(
    tables: &[TabularData],
    posterior: &GraphPosterior,
    prior: &OutcomePriorSource,
) -> ChainReport {
    let mut points = Vec::new();
    let mut variances = Vec::new();
    let mut unanswered = 0.0;
    let mut sample = None;
    for (k, table) in tables.iter().enumerate() {
        // (mass, effect, standard error) per answerable structure.
        let mut parts: Vec<(f64, f64, f64)> = Vec::new();
        let mut dropped = 0.0;
        for (index, (atom, &weight)) in
            posterior.atoms().iter().zip(posterior.weights().iter()).enumerate()
        {
            if weight < NEGLIGIBLE_MASS {
                dropped += weight;
            } else if !reaches(atom, X, Y) {
                // Without a directed path from the exposure the effect is exactly zero.
                parts.push((weight, 0.0, 0.0));
            } else if let Ok((ate, se, result)) =
                estimate_structure(table, atom, prior, 1_000 + (k * 64 + index) as u64)
            {
                if sample.is_none() {
                    sample = Some(result);
                }
                parts.push((weight, ate, se));
            } else {
                dropped += weight;
            }
        }
        unanswered = dropped;
        let answered: f64 = parts.iter().map(|part| part.0).sum();
        let mixed: f64 = parts.iter().map(|(w, ate, _)| w * ate).sum::<f64>() / answered;
        let within = parts.iter().map(|(w, _, se)| w * se).sum::<f64>() / answered;
        let between =
            parts.iter().map(|(w, ate, _)| w * (ate - mixed).powi(2)).sum::<f64>() / answered;
        points.push(mixed);
        variances.push(within * within + between);
    }
    ChainReport {
        effect: pool_measured_effect(&points, &variances, None).unwrap(),
        unanswered_mass: unanswered,
        sample: sample.expect("at least one structure answers the declared target"),
    }
}

// ---- the naive pipeline -------------------------------------------------------------------

/// The recorded column taken at face value, one assumed structure, no prior, every row.
fn naive_effect(planted: &Planted) -> f64 {
    let recorded: Vec<f64> = planted.recorded.iter().copied().map(flag).collect();
    let table = TabularData::from_f64_columns([
        ("z", planted.z.as_slice()),
        ("x", recorded.as_slice()),
        ("y", planted.y.as_slice()),
    ])
    .unwrap();
    let mut graph = Dag::with_variables(3);
    for (from, to) in [(Z, X), (Z, Y), (X, Y)] {
        graph
            .insert_directed(DenseNodeId::from_raw(from as u32), DenseNodeId::from_raw(to as u32))
            .unwrap();
    }
    let result = Study::tabular(table)
        .graph(graph)
        .estimator(AipwAte::new().with_bootstrap_replicates(0))
        .refute(RefuteSuite::None)
        .query(AverageEffectQuery::binary_ate(v(X), v(Y)))
        .build()
        .unwrap()
        .run(&ctx(7))
        .unwrap();
    result.estimate().as_effect().unwrap().ate
}

#[test]
fn the_chain_recovers_the_closed_form_population_effect_and_names_every_stage() {
    on_large_stack(|| {
        let planted = plant(2_026);
        let truth = closed_form_truth(&planted);

        // Stage 1: the measured exposure, declared with a validation subsample.
        let measured = StudyBuilder::measured_exposure(
            &declaration(&planted),
            columns(&planted),
            MeasuredExposureRequest {
                exposure: "x".into(),
                outcome: "y".into(),
                covariates: vec!["z".into()],
                draws: DRAWS,
                seed: 31,
            },
            &ctx(31),
        )
        .unwrap();
        let draws = measured.draws();
        assert_eq!(draws.draw_count, DRAWS);
        assert!(draws.draws.iter().all(|&value| value == 0.0 || value == 1.0));
        assert!(!draws.declaration_identity.is_empty());
        let tables: Vec<TabularData> =
            (0..DRAWS).map(|k| measured.completed_table(k).unwrap()).collect();
        assert_eq!(antecedent::data::TableView::row_count(&tables[0]), ROWS);

        // Stage 2: the structure search over the completed tables, pooled in the engine.
        let params = structure_params();
        let posterior = antecedent::discovery::discover_pooled_exact_dag_posterior(
            &tables,
            &three(),
            &params,
            &ctx(32),
        )
        .unwrap();
        assert_eq!(
            posterior.algorithm().map(|algorithm| &**algorithm),
            Some("pooled_imputations(exact_dag_posterior)")
        );
        let total: f64 = posterior.weights().iter().sum();
        assert!((total - 1.0).abs() < 1e-9, "{total}");
        // The declared prior breaks the tie the data cannot: the exposure points at the outcome.
        let with_prior = mass_of_edge(&posterior, X, Y);
        let without_prior = mass_of_edge(
            &antecedent::discovery::discover_pooled_exact_dag_posterior(
                &tables,
                &three(),
                &BayesianDiscoverParams::default(),
                &ctx(32),
            )
            .unwrap(),
            X,
            Y,
        );
        assert!(with_prior > 0.9, "x -> y holds {with_prior} of the prior-informed posterior");
        assert!(
            with_prior > without_prior + 0.1,
            "the prior moved x -> y from {without_prior} to {with_prior}"
        );
        let statement = structure_prior_statement(&params);
        assert!(statement.contains("v1->v2:0.5"), "{statement}");

        // Stage 3 and 4: the evidence-weighted, population-targeted effect over the posterior.
        let composed = composed_prior();
        let outcome_prior = outcome_prior_source_from_composed(&composed).unwrap();
        let report = composed_effect(&tables, &posterior, &outcome_prior);
        let chain = report.effect.pooled.point;
        let chain_se = report.effect.standard_error();
        assert!(report.unanswered_mass < 0.05, "unanswered mass {}", report.unanswered_mass);
        assert!(
            (chain - truth).abs() < TOLERANCE,
            "the chain reports {chain} (se {chain_se}) against the closed-form {truth}"
        );

        // The chain matters: the naive pipeline misses the truth by more than the tolerance.
        let naive = naive_effect(&planted);
        assert!(
            (naive - truth).abs() > TOLERANCE,
            "the naive pipeline reports {naive}, near the closed-form {truth}"
        );
        assert!((chain - truth).abs() < (naive - truth).abs());

        // Provenance: the estimate's own record carries the composed prior; the declared target
        // and the earlier stages are named by the objects they produced.
        let effect = report.sample.estimate().as_effect().unwrap();
        let prior_record = effect
            .provenance
            .assumptions
            .entries
            .iter()
            .find_map(|record| match &record.assumption {
                Assumption::PriorRestriction(prior)
                    if &*prior.id == antecedent_estimate::OUTCOME_PRIOR_ASSUMPTION_ID =>
                {
                    Some(prior.description.to_string())
                }
                _ => None,
            })
            .expect("the outcome nuisance prior is a recorded assumption");
        assert!(prior_record.contains("composed:blake3:"), "{prior_record}");
        assert!(prior_record.contains("intercept, coef_z"), "{prior_record}");
        let mut triangle = Dag::with_variables(3);
        for (from, to) in [(Z, X), (Z, Y), (X, Y)] {
            triangle
                .insert_directed(
                    DenseNodeId::from_raw(from as u32),
                    DenseNodeId::from_raw(to as u32),
                )
                .unwrap();
        }
        let prepared = Study::tabular(tables[0].clone())
            .graph(triangle)
            .estimator(AipwAte::new().with_bootstrap_replicates(0))
            .refute(RefuteSuite::None)
            .weighted_target(target())
            .query(AverageEffectQuery::binary_ate(v(X), v(Y)))
            .build()
            .unwrap()
            .prepare(&ctx(33))
            .unwrap();
        let target_report = prepared.weighted_target_report().expect("the target is reported");
        assert_eq!(target_report.family(), Some(KernelFamily::Gaussian));
        assert_eq!(target_report.center(), Some(&[CENTRE][..]));
        assert!(!target_report.identity().is_empty());
        assert!(!draws.declaration_identity.is_empty() && !draws.data_identity.is_empty());
    });
}
