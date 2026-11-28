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
//!    is pooled across them in the engine, returning an accepted posterior with its provenance;
//! 3. on every completed table a study over that posterior estimates the effect by cross-fitted
//!    AIPW in every structure the posterior supports, with the outcome nuisance regularised by an
//!    external prior whose power-prior discount is selected from the table itself, and mixes the
//!    structure-wise effects by posterior weight with the between-structure variance;
//! 4. the estimate is for a declared kernel-weighted target population over `z`, whose truth is
//!    the closed-form kernel-weighted mean of the planted modified effect on the planted rows;
//!    the per-table mixtures are pooled by Rubin's rules.
//!
//! The planted effect is `tau(z) = 1.5 + 0.5 z`. The stages hand each other only the objects the
//! library produced: no atom loop, no pooling and no provenance is composed by the caller. The
//! pooled result names every stage it consumed.
//!
//! The naive pipeline (the recorded column, one assumed structure, no prior, every row) misses the
//! truth by more than the tolerance, so each stage earns its place.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::too_many_lines,
    reason = "integration test with a planted truth"
)]

mod common;

use std::sync::Arc;

use antecedent::discovery::{
    BayesianDiscoverParams, EdgePriors, GraphPosterior, GraphPrior, SoftTiers,
};
use antecedent::estimate::measurement::{ExposureErrorDeclaration, ExposureMeasurement, RatePair};
use antecedent::inference::{AlphaCriterion, AlphaSelectionOptions, KernelFamily, TransportKernel};
use antecedent::measured_exposure::{MeasuredColumns, MeasuredExposureRequest};
use antecedent::{
    AverageEffectQuery, DiscoverySpec, OutcomePriorSelection, RefuteSuite, Study, StudyBuilder,
    WeightedTargetDeclaration,
};
use antecedent_core::{ExecutionContext, VariableId};
use antecedent_data::TabularData;
use antecedent_estimate::AipwAte;
use antecedent_graph::{Dag, DenseNodeId};
use antecedent_prob::{
    ExternalPriorSource, ExternalPriorWeight, GaussianCoefficientPrior, PriorSet, PriorSpec,
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
    antecedent::support::caller_thread()
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

/// Two external sources on the outcome regression; the study selects their common discount
/// from each completed table.
fn prior_selection() -> OutcomePriorSelection {
    OutcomePriorSelection::new(
        vec![
            external("registry-a", [1.6, 0.7], [0.6, 0.5], 1.0),
            external("trial-b", [1.9, 0.8], [0.8, 0.6], 1.0),
        ],
        gaussian_prior(vec![0.0; 2], vec![100.0; 2]),
        names(&["intercept", "coef_z"]),
        AlphaSelectionOptions::new(AlphaCriterion::EmpiricalBayes),
    )
    .unwrap()
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
        assert!(!draws.declaration_identity.is_empty() && !draws.data_identity.is_empty());
        assert_eq!(
            antecedent::data::TableView::row_count(&measured.completed_table(0).unwrap()),
            ROWS
        );

        // Stage 2: the structure search on the completed tables, pooled in the engine and
        // accepted with its provenance, exactly as a single compiled run's posterior is.
        let params = structure_params();
        let spec = DiscoverySpec::exact_dag_posterior(params.clone());
        let accepted = measured.structure_posterior(&spec, &three(), &ctx(32)).unwrap();
        let posterior = accepted.posterior();
        assert_eq!(
            posterior.algorithm().map(|algorithm| &**algorithm),
            Some("pooled_imputations(exact_dag_posterior)")
        );
        let total: f64 = posterior.weights().iter().sum();
        assert!((total - 1.0).abs() < 1e-9, "{total}");
        let uniform_posterior = measured
            .structure_posterior(
                &DiscoverySpec::exact_dag_posterior(BayesianDiscoverParams::default()),
                &three(),
                &ctx(32),
            )
            .unwrap();
        // The declared prior breaks the tie the data cannot: the exposure points at the outcome.
        let with_prior = mass_of_edge(posterior, X, Y);
        let without_prior = mass_of_edge(uniform_posterior.posterior(), X, Y);
        assert!(with_prior > 0.9, "x -> y holds {with_prior} of the prior-informed posterior");
        assert!(
            with_prior > without_prior + 0.1,
            "the prior moved x -> y from {without_prior} to {with_prior}"
        );
        let provenance = accepted.provenance();
        let statement = provenance.structure_prior.as_deref().expect("the declared prior");
        assert_eq!(statement, structure_prior_statement(&params));
        assert!(statement.contains("v1->v2:0.5"), "{statement}");
        let pooling = provenance.pooling.as_ref().expect("the pooling over imputations");
        assert_eq!(pooling.imputations, DRAWS as u64);
        assert!(pooling.source.contains(&draws.declaration_identity), "{}", pooling.source);

        // Stages 3 and 4: on every completed table a study over the accepted posterior
        // estimates the declared population's effect in every structure and mixes them; the
        // per-table mixtures are pooled by Rubin's rules. Nothing is looped or pooled here.
        let selection = prior_selection();
        let chain = measured
            .estimate_over_structures(&spec, &three(), &ctx(32), |table, accepted| {
                Study::tabular(table.clone())
                    .accepted_graph_posterior(accepted.clone())
                    .estimator(AipwAte::new().with_bootstrap_replicates(0))
                    .outcome_prior_selection(selection.clone())
                    .refute(RefuteSuite::None)
                    .weighted_target(target())
                    .query(AverageEffectQuery::binary_ate(v(X), v(Y)))
                    .build()?
                    .run(&ctx(33))
            })
            .unwrap();
        let pooled = chain.effect.pooled.point;
        let pooled_se = chain.effect.standard_error();
        assert!(chain.unanswered_mass < 0.05, "unanswered mass {}", chain.unanswered_mass);
        assert!(
            (pooled - truth).abs() < TOLERANCE,
            "the chain reports {pooled} (se {pooled_se}) against the closed-form {truth}"
        );

        // The chain matters: the naive pipeline misses the truth by more than the tolerance.
        let naive = naive_effect(&planted);
        assert!(
            (naive - truth).abs() > TOLERANCE,
            "the naive pipeline reports {naive}, near the closed-form {truth}"
        );
        assert!((pooled - truth).abs() < (naive - truth).abs());

        // Provenance: the pooled result names every stage it consumed.
        assert_eq!(chain.declaration_identity, draws.declaration_identity);
        assert!(
            chain.provenance.measurement.contains(&draws.declaration_identity)
                && chain.provenance.measurement.contains(&draws.data_identity),
            "{}",
            chain.provenance.measurement
        );
        let structure = chain.provenance.structure.as_ref().expect("the structure stage");
        assert_eq!(structure.structure_prior, provenance.structure_prior);
        assert_eq!(structure.pooling, provenance.pooling);
        let composed = chain
            .provenance
            .declaration(antecedent_estimate::OUTCOME_PRIOR_ASSUMPTION_ID)
            .expect("the outcome nuisance prior is a recorded stage");
        assert!(composed.contains("composed:blake3:"), "{composed}");
        assert!(composed.contains("intercept, coef_z"), "{composed}");
        let selected = chain
            .provenance
            .declarations
            .iter()
            .find(|(id, _)| id.starts_with("power_prior_alpha_selected:criterion=empirical_bayes"))
            .expect("the data-selected discount is a recorded stage");
        assert!(selected.1.contains("selected from the target data"), "{}", selected.1);
        let target_statement = chain
            .provenance
            .declaration("estimate.weighted_target_population")
            .expect("the weighted target is a recorded stage");
        assert!(target_statement.contains("kernel neighbourhood"), "{target_statement}");
        let structure_statement = chain
            .provenance
            .declaration(antecedent::STRUCTURE_STAGE_ASSUMPTION_ID)
            .expect("the structure posterior is a recorded stage");
        assert!(
            structure_statement.contains("v1->v2:0.5")
                && structure_statement.contains(&format!("pooled over {DRAWS} imputations")),
            "{structure_statement}"
        );
        assert!(chain.provenance.statements().len() >= 6);
    });
}
