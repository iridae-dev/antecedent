//! A structure posterior pooled over completed tables is accepted with its provenance, and a
//! power-prior discount is selected from the study's own table.
//!
//! * The pooled accepted posterior is the engine's equal-weight mixture of the per-table
//!   posteriors, atom for atom, and a study built on it carries the search, its declared structure
//!   prior and the pooling over imputations in its contract exactly as a single compiled run's
//!   posterior does.
//! * The selection run by a study equals the selection run by hand on the same design, a
//!   discrepant source is discounted below a concordant one, and the selection is recorded in the
//!   assumptions of the result; with a diffuse prior the prior-regularized estimate for a weighted
//!   target equals the unregularized one.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    deprecated,
    reason = "these tests call the direct discovery functions the compiled route is compared with"
)]
#![allow(clippy::cast_possible_truncation, reason = "small graph node counts")]
#![allow(
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::too_many_lines,
    reason = "integration test with a planted truth and a hand-composed reference"
)]
mod common;

use std::sync::Arc;

use antecedent::discovery::{
    BayesianDiscoverParams, EdgePriors, GraphPrior, discover_pooled_exact_dag_posterior,
};
use antecedent::inference::{
    AlphaCriterion, AlphaSelectionOptions, KernelFamily, LinearTarget, TransportKernel,
    select_linear_prior,
};
use antecedent::{
    AverageEffectQuery, DiscoverySpec, OutcomePriorSelection, RefuteSuite, Study,
    WeightedTargetDeclaration,
};
use antecedent_core::{Assumption, ExecutionContext, VariableId};
use antecedent_data::{TableView, TabularData};
use antecedent_estimate::AipwAte;
use antecedent_graph::{Dag, DenseNodeId};
use antecedent_prob::{
    ExternalPriorSource, ExternalPriorWeight, GaussianCoefficientPrior, PriorSet, PriorSpec,
};
use common::calibration::{gaussian, uniform};

const ROWS: usize = 2_000;
const TABLES: usize = 4;

fn v(raw: usize) -> VariableId {
    VariableId::from_raw(u32::try_from(raw).unwrap())
}

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(5)
}

fn variables() -> Vec<VariableId> {
    vec![v(0), v(1), v(2)]
}

/// `z -> t -> y` and `z -> y`, with the effect of `t` equal to `1 + 0.5 z`.
fn table(seed: u64) -> TabularData {
    let mut normal = gaussian(seed);
    let mut next = uniform(seed, 1);
    let z: Vec<f64> = (0..ROWS).map(|_| normal()).collect();
    let t: Vec<f64> =
        z.iter().map(|&zi| f64::from(u8::from(next() < 1.0 / (1.0 + (-0.8 * zi).exp())))).collect();
    let y: Vec<f64> =
        (0..ROWS).map(|i| 0.5 * z[i] + t[i] * (1.0 + 0.5 * z[i]) + 0.5 * normal()).collect();
    TabularData::from_f64_columns([("z", z.as_slice()), ("t", t.as_slice()), ("y", y.as_slice())])
        .unwrap()
}

fn tables() -> Vec<TabularData> {
    (0..TABLES as u64).map(|k| table(300 + k)).collect()
}

fn params() -> BayesianDiscoverParams {
    BayesianDiscoverParams {
        prior: GraphPrior::uniform()
            .with_edge_priors(EdgePriors::new(vec![(v(1), v(2), 0.5), (v(2), v(1), 0.02)])),
        ..BayesianDiscoverParams::default()
    }
}

fn dag(edges: &[(usize, usize)]) -> Dag {
    let mut dag = Dag::with_variables(3);
    for &(a, b) in edges {
        dag.insert_directed(DenseNodeId::from_raw(a as u32), DenseNodeId::from_raw(b as u32))
            .unwrap();
    }
    dag
}

// ---- pooled provenance --------------------------------------------------------------------

#[test]
fn the_pooled_accepted_posterior_is_the_engine_mixture_with_its_provenance() {
    let tables = tables();
    let spec = DiscoverySpec::exact_dag_posterior(params());
    let accepted = spec
        .pooled_posterior(&tables, &variables(), Some("a measurement declaration"), &ctx())
        .unwrap();
    let engine =
        discover_pooled_exact_dag_posterior(&tables, &variables(), &params(), &ctx()).unwrap();
    assert_eq!(accepted.posterior().atoms().as_ref(), engine.atoms().as_ref());
    assert_eq!(accepted.posterior().weights().as_ref(), engine.weights().as_ref());

    let provenance = accepted.provenance();
    assert_eq!(&*provenance.method, "exact_dag_posterior");
    assert_eq!(provenance.review.as_ref(), "posterior");
    let prior = provenance.structure_prior.as_deref().expect("the declared structure prior");
    assert!(prior.contains("v1->v2:0.5"), "{prior}");
    let pooling = provenance.pooling.as_ref().expect("the pooling over imputations");
    assert_eq!(pooling.imputations, TABLES as u64);
    assert_eq!(&*pooling.source, "a measurement declaration");
    assert_eq!(&*pooling.method, "equal_weight_mixture_of_per_imputation_posteriors");

    // The identity changes with the pooling, the source and the tables, and is reproducible.
    let again = spec
        .pooled_posterior(&tables, &variables(), Some("a measurement declaration"), &ctx())
        .unwrap();
    assert_eq!(again.provenance().digest(), provenance.digest());
    let other_source =
        spec.pooled_posterior(&tables, &variables(), Some("another declaration"), &ctx()).unwrap();
    assert_ne!(other_source.provenance().digest(), provenance.digest());
    let fewer = spec
        .pooled_posterior(&tables[..2], &variables(), Some("a measurement declaration"), &ctx())
        .unwrap();
    assert_ne!(fewer.provenance().digest(), provenance.digest());
    let mut unpooled = provenance.clone();
    unpooled.pooling = None;
    assert_ne!(unpooled.digest(), provenance.digest());
    assert!(provenance.statement().contains("pooled over 4 imputations"));
}

#[test]
fn a_study_on_the_pooled_posterior_carries_the_provenance_like_a_compiled_one() {
    let tables = tables();
    let spec = DiscoverySpec::exact_dag_posterior(params());
    let accepted = spec.pooled_posterior(&tables, &variables(), None, &ctx()).unwrap();
    let digest = accepted.provenance().digest();
    let study = Study::tabular(tables[0].clone())
        .accepted_graph_posterior(accepted.clone())
        .refute(RefuteSuite::None)
        .query(AverageEffectQuery::binary_ate(v(1), v(2)))
        .build()
        .unwrap();
    let contract = study.prepare(&ctx()).unwrap().inspect().unwrap();
    let carried = contract.structure_provenance.as_ref().expect("the structure provenance");
    assert_eq!(carried.digest(), digest);
    assert_eq!(carried.pooling, accepted.provenance().pooling);
    // A study on the same atoms supplied by hand carries none.
    let plain = Study::tabular(tables[0].clone())
        .graph_posterior(accepted.posterior().clone())
        .refute(RefuteSuite::None)
        .query(AverageEffectQuery::binary_ate(v(1), v(2)))
        .build()
        .unwrap();
    assert!(plain.prepare(&ctx()).unwrap().inspect().unwrap().structure_provenance.is_none());
}

#[test]
fn a_method_that_lists_no_graphs_does_not_pool() {
    let spec = DiscoverySpec::exact_dag_marginals(params());
    assert!(spec.pooled_posterior(&tables(), &variables(), None, &ctx()).is_err());
}

// ---- alpha selection in the study ---------------------------------------------------------

fn gaussian_prior(mean: Vec<f64>, variance: Vec<f64>) -> PriorSet {
    let mut prior = PriorSet::new();
    prior.push(PriorSpec::GaussianCoefficients(GaussianCoefficientPrior {
        mean: Arc::from(mean),
        variance: Arc::from(variance),
    }));
    prior
}

fn source(id: &str, mean: [f64; 2], variance: [f64; 2]) -> ExternalPriorSource {
    ExternalPriorSource {
        id: Arc::from(id),
        prior: gaussian_prior(mean.to_vec(), variance.to_vec()),
        weight: ExternalPriorWeight::power(1.0).unwrap(),
        ess: None,
    }
}

fn names() -> Vec<Arc<str>> {
    vec![Arc::from("intercept"), Arc::from("coef_z")]
}

fn selection(sources: Vec<ExternalPriorSource>) -> OutcomePriorSelection {
    OutcomePriorSelection::new(
        sources,
        gaussian_prior(vec![0.0; 2], vec![100.0; 2]),
        names(),
        AlphaSelectionOptions::new(AlphaCriterion::EmpiricalBayes),
    )
    .unwrap()
}

/// The pooled regression of `y` on `(1, z)` has intercept about 0.5 and slope about 0.75.
fn concordant() -> ExternalPriorSource {
    source("concordant", [0.5, 0.75], [0.01, 0.01])
}

fn discrepant() -> ExternalPriorSource {
    source("discrepant", [6.0, -4.0], [0.01, 0.01])
}

#[test]
fn the_study_selection_equals_the_selection_run_by_hand_on_the_same_design() {
    let data = table(77);
    let resolved = selection(vec![concordant()]).resolve(&data, v(2)).unwrap();

    let z = data.float64_values(v(0)).unwrap();
    let y = data.float64_values(v(2)).unwrap();
    let design: Vec<f64> = z.iter().flat_map(|&zi| [1.0, zi]).collect();
    let mean_z = z.iter().sum::<f64>() / z.len() as f64;
    let (composed, report) = select_linear_prior(
        &[concordant()],
        &gaussian_prior(vec![0.0; 2], vec![100.0; 2]),
        LinearTarget { x: &design, y: &y, n_rows: z.len(), n_cols: 2, functional: &[1.0, mean_z] },
        &AlphaSelectionOptions::new(AlphaCriterion::EmpiricalBayes),
    )
    .unwrap();
    assert_eq!(resolved.report.selected_alpha, report.selected_alpha);
    assert_eq!(resolved.report.provenance_id(), report.provenance_id());
    assert_eq!(resolved.composed.prior, composed.prior);
    assert_eq!(resolved.composed.coefficient_names.as_deref().map(<[_]>::len), Some(2));
}

#[test]
fn a_discrepant_source_is_discounted_below_a_concordant_one() {
    let data = table(78);
    let kept = selection(vec![concordant()]).resolve(&data, v(2)).unwrap();
    let dropped = selection(vec![discrepant()]).resolve(&data, v(2)).unwrap();
    assert!(
        dropped.report.selected_alpha < kept.report.selected_alpha,
        "discrepant {} against concordant {}",
        dropped.report.selected_alpha,
        kept.report.selected_alpha
    );
    // A coefficient that names no column of the table is refused.
    let bad = OutcomePriorSelection::new(
        vec![concordant()],
        gaussian_prior(vec![0.0; 2], vec![100.0; 2]),
        vec![Arc::from("intercept"), Arc::from("coef_missing")],
        AlphaSelectionOptions::new(AlphaCriterion::EmpiricalBayes),
    )
    .unwrap();
    assert!(bad.resolve(&data, v(2)).is_err());
}

fn target() -> WeightedTargetDeclaration {
    WeightedTargetDeclaration::kernel(
        TransportKernel::new(KernelFamily::Gaussian, 0.8).unwrap(),
        vec![v(0)],
        vec![0.3],
        None,
    )
    .unwrap()
}

fn effect(study: &Study) -> (f64, Vec<String>) {
    let result = study.run(&ctx()).unwrap();
    let effect = result.estimate().as_effect().unwrap();
    let ids = effect
        .provenance
        .assumptions
        .entries
        .iter()
        .filter_map(|record| match &record.assumption {
            Assumption::PriorRestriction(declared) => Some(declared.id.to_string()),
            _ => None,
        })
        .collect();
    (effect.ate, ids)
}

#[test]
fn the_selection_is_recorded_and_a_diffuse_prior_leaves_the_weighted_estimate_unchanged() {
    let data = table(79);
    let build = || {
        Study::tabular(data.clone())
            .graph(dag(&[(0, 1), (0, 2), (1, 2)]))
            .refute(RefuteSuite::None)
            .weighted_target(target())
            .query(AverageEffectQuery::binary_ate(v(1), v(2)))
    };
    let (plain, plain_ids) =
        effect(&build().estimator(AipwAte::new().with_bootstrap_replicates(0)).build().unwrap());
    assert!(plain_ids.iter().all(|id| !id.starts_with("power_prior_alpha_selected")));

    // Sources so diffuse that any discount leaves the regression where the data put it.
    let diffuse = selection(vec![
        source("diffuse-a", [0.0, 0.0], [1e12, 1e12]),
        source("diffuse-b", [1.0, 1.0], [1e12, 1e12]),
    ]);
    let (regularized, ids) = effect(
        &build()
            .estimator(AipwAte::new().with_bootstrap_replicates(0))
            .outcome_prior_selection(diffuse)
            .build()
            .unwrap(),
    );
    assert!(
        ids.iter().any(|id| id.starts_with("power_prior_alpha_selected:criterion=empirical_bayes")),
        "{ids:?}"
    );
    assert!(ids.iter().any(|id| id == antecedent_estimate::OUTCOME_PRIOR_ASSUMPTION_ID), "{ids:?}");
    assert!(
        (regularized - plain).abs() < 1e-3,
        "the diffuse-prior estimate {regularized} against the unregularized {plain}"
    );
    assert_eq!(antecedent::data::TableView::row_count(&data), ROWS);
}

#[test]
fn a_selection_beside_another_prior_or_estimator_is_refused() {
    let data = table(80);
    let refused = Study::tabular(data)
        .graph(dag(&[(0, 1), (0, 2), (1, 2)]))
        .refute(RefuteSuite::None)
        .outcome_prior_selection(selection(vec![concordant()]))
        .query(AverageEffectQuery::binary_ate(v(1), v(2)))
        .build();
    assert!(refused.is_err(), "no AipwAte or DmlAte estimator carries the selected prior");
}
