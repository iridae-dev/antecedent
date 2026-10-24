//! Associational lowering: both cells lower completely, the identification layer claims
//! nothing, the retained program fixes the selected analysis, and the declared sampling design
//! decides the resampling of a series.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss, reason = "fixture indices convert exactly")]

use antecedent_contract::ClaimKind;
use antecedent_core::{
    AssociationFunctional, AssociationQuery, CausalQuery, ExecutionContext, VariableId,
};
use antecedent_data::{TabularData, TimeSeriesData};

use super::{lowered_analysis, retained_refusal};
use crate::analysis::lowering::registry::REGISTERED;
use crate::planner::lowered::{FunctionalKind, IdentificationLayer, LoweredProgram, ProblemTarget};
use crate::planner::statistical::{
    PosteriorAlgorithm, Realization, ResampleUnit, ResamplingMethod,
};
use crate::{BayesianConfig, InferenceMode, Study};

fn v(raw: u32) -> VariableId {
    VariableId::from_raw(raw)
}

fn compiled(study: &Study) -> LoweredProgram {
    let plan = study.compile(&ExecutionContext::for_tests(9)).unwrap();
    crate::planner::plan_view::checks::assert_view_agrees("association", &plan);
    plan.lowered().expect("compile attaches a lowered program").clone()
}

fn table() -> TabularData {
    let x: Vec<f64> = (0..40).map(|i| f64::from(i) / 4.0).collect();
    let y: Vec<f64> =
        x.iter().enumerate().map(|(i, x)| 1.0 + x + ((i * 7) % 5) as f64 * 0.1).collect();
    TabularData::from_f64_columns([("x", &x[..]), ("y", &y[..])]).unwrap()
}

fn series() -> TimeSeriesData {
    let x: Vec<f64> = (0..60).map(|i| ((i * 13) % 11) as f64).collect();
    let y: Vec<f64> = x.iter().enumerate().map(|(i, x)| 2.0 * x + (i % 3) as f64).collect();
    TimeSeriesData::from_f64_columns([("x", &x[..]), ("y", &y[..])], 1_000).unwrap()
}

fn study(query: AssociationQuery) -> Study {
    Study::tabular(table()).query(CausalQuery::Association(query)).build().unwrap()
}

#[test]
fn every_association_cell_is_lowered_by_its_family() {
    let cells: Vec<_> = REGISTERED
        .iter()
        .filter(|(_, family)| *family == "associational")
        .map(|(cell, _)| cell.as_str())
        .collect();
    assert_eq!(cells, ["AssociationStatic", "AssociationTemporal"]);
}

#[test]
fn an_association_claims_no_identification_and_keeps_the_refusal_it_follows() {
    let plain = compiled(&study(AssociationQuery::new(v(1), [v(0)])));
    assert_eq!(plain.route(), "AssociationStatic");
    assert!(matches!(
        plain.scientific().target(),
        ProblemTarget::QueryFunctional { kind: FunctionalKind::Association, .. }
    ));
    assert!(matches!(
        plain.identification_layer(),
        IdentificationLayer::NotClaimed { refusal: None }
    ));
    assert!(!plain.identification_retained());
    assert_eq!(plain.claim().kind(), Some(ClaimKind::Associational));
    assert_eq!(lowered_analysis(&plain), Some("linear_regression"));
    assert_eq!(retained_refusal(&plain), None);
    let refused = compiled(&study(
        AssociationQuery::new(v(1), [v(0)]).following_refusal("effect_not_identified"),
    ));
    assert_eq!(retained_refusal(&refused).as_deref(), Some("effect_not_identified"));
    assert_ne!(refused.identity(), plain.identity(), "the retained refusal enters the identity");
    let stable = compiled(&study(AssociationQuery::new(v(1), [v(0)])));
    assert_eq!(stable.identity(), plain.identity(), "the identity is stable");
}

#[test]
fn the_program_fixes_the_analysis_the_selection_rule_picked() {
    let conditional = compiled(&study(
        AssociationQuery::new(v(1), [v(0)]).with_functional(AssociationFunctional::Conditional),
    ));
    assert_eq!(lowered_analysis(&conditional), Some("stratified_mean"));
    let predictive = compiled(&study(
        AssociationQuery::new(v(1), [v(0)])
            .with_functional(AssociationFunctional::Predictive { folds: 4 }),
    ));
    assert_eq!(lowered_analysis(&predictive), Some("predictive_linear"));
    let folds = predictive.statistical().folds().expect("a predictive model has held-out folds");
    assert_eq!((folds.folds(), folds.unit()), (4, ResampleUnit::Row));
    assert_ne!(conditional.identity(), predictive.identity());
}

#[test]
fn a_series_resamples_blocks_and_a_bayesian_reading_declares_its_posterior() {
    let data = series();
    let frequentist = Study::series(data.clone())
        .query(CausalQuery::Association(AssociationQuery::new(v(1), [v(0)])))
        .bootstrap_replicates(30)
        .build()
        .unwrap();
    let program = compiled(&frequentist);
    assert_eq!(program.route(), "AssociationTemporal");
    let resampling = program.statistical().resampling().expect("replicates were requested");
    assert_eq!(resampling.method(), ResamplingMethod::BlockBootstrap);
    assert_eq!(resampling.unit(), ResampleUnit::CalendarBlock);
    assert!(matches!(program.statistical().realization(), Realization::Frequentist(_)));
    let bayesian = Study::series(data)
        .query(CausalQuery::Association(AssociationQuery::new(v(1), [v(0)])))
        .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(100)))
        .build()
        .unwrap();
    let program = compiled(&bayesian);
    assert!(matches!(program.statistical().realization(), Realization::Bayesian(_)));
    assert!(program.statistical().resampling().is_none());
    assert_eq!(
        program
            .statistical()
            .posterior()
            .map(crate::planner::statistical::PosteriorPlan::algorithm),
        Some(PosteriorAlgorithm::BayesianBootstrap)
    );
}
