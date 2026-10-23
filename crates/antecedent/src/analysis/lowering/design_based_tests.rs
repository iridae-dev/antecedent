//! Design-based lowering: every cell of the family lowers completely, and a compiled design
//! carries its own target, retained design identification, statistical program and identity.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    reason = "the fixture's 53-bit integer state converts exactly"
)]

use std::sync::Arc;

use antecedent_core::{
    CausalQuery, ExecutionContext, GraphFamily, HistoryMatchQuery, SequentialContrastQuery,
    TargetForm, VariableId,
};
use antecedent_data::TabularData;

use super::{LOWERED, retained_assumptions};
use crate::analysis::lowering::registry::REGISTERED;
use crate::planner::lowered::{
    FunctionalKind, LoweredProgram, LoweringDepth, ProblemTarget, WidthSource,
};
use crate::planner::statistical::{Realization, ResampleUnit};
use crate::{BayesianConfig, EstimatorId, InferenceMode, Study};

const SUBJECTS: usize = 80;
const DRAWS: usize = 400;

#[test]
fn every_design_cell_is_lowered_by_its_family() {
    let family: Vec<_> = REGISTERED
        .iter()
        .filter(|(_, family)| *family == "design_based")
        .map(|(cell, _)| *cell)
        .collect();
    assert_eq!(family.len(), LOWERED.len(), "the family's cell inventory drifted");
    for cell in family {
        assert!(LOWERED.contains(&cell), "{} is not lowered by the design family", cell.as_str());
    }
}

fn compiled(study: &Study) -> LoweredProgram {
    let ctx = ExecutionContext::for_tests(5);
    let plan = study.compile(&ctx).unwrap();
    plan.lowered().expect("compile attaches a lowered program").clone()
}

fn recorded_estimator(study: &Study) -> Option<EstimatorId> {
    let ctx = ExecutionContext::for_tests(5);
    let plan = study.compile(&ctx).unwrap();
    plan.logical.record.estimator.as_deref().and_then(|name| name.parse::<EstimatorId>().ok())
}

fn uniform(state: &mut u64) -> f64 {
    *state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
    (*state >> 11) as f64 / 9_007_199_254_740_992.0
}

/// Columns l0, a0, l1, a1, y, in that order.
fn panel() -> TabularData {
    let mut state = 12_345_u64;
    let (mut l0s, mut a0s, mut l1s, mut a1s, mut ys) = (vec![], vec![], vec![], vec![], vec![]);
    for _ in 0..SUBJECTS {
        let u1 = uniform(&mut state);
        let u2 = uniform(&mut state);
        let u3 = uniform(&mut state);
        let u4 = uniform(&mut state);
        let u5 = uniform(&mut state);
        let l0 = 2.0 * u1 - 1.0;
        let a0 = if u2 < 0.5 + 0.2 * l0 { 1.0 } else { 0.0 };
        let l1 = 0.5 * l0 + 0.8 * a0 + (2.0 * u3 - 1.0) * 0.5;
        let a1 = if u4 < 0.5 + 0.1 * (l1 - 0.4) { 1.0 } else { 0.0 };
        ys.push(1.0 + 0.5 * l0 + 1.0 * a0 + 1.5 * a1 + 0.4 * l1 + (2.0 * u5 - 1.0));
        l0s.push(l0);
        a0s.push(a0);
        l1s.push(l1);
        a1s.push(a1);
    }
    TabularData::from_f64_columns([
        ("l0", l0s.as_slice()),
        ("a0", a0s.as_slice()),
        ("l1", l1s.as_slice()),
        ("a1", a1s.as_slice()),
        ("y", ys.as_slice()),
    ])
    .unwrap()
}

fn contrast_query(active: [u8; 2]) -> SequentialContrastQuery {
    SequentialContrastQuery {
        outcome: VariableId::from_raw(4),
        first_treatment: VariableId::from_raw(1),
        second_treatment: VariableId::from_raw(3),
        baseline: Arc::from([VariableId::from_raw(0)]),
        intermediate: Arc::from([VariableId::from_raw(2)]),
        active,
        control: [0, 0],
        fold_of_row: (0..SUBJECTS).map(|row| u32::from(matches!(row % 4, 1 | 2))).collect(),
        fold_count: 2,
        min_probability: 0.05,
    }
}

fn sequential_study(
    data: &TabularData,
    query: SequentialContrastQuery,
    inference: InferenceMode,
) -> Study {
    Study::tabular(data.clone())
        .query(CausalQuery::SequentialContrast(query))
        .inference(inference)
        .build()
        .unwrap()
}

#[test]
fn sequential_contrast_lowers_its_regime_contrast_and_cross_fit() {
    let data = panel();
    let study = sequential_study(&data, contrast_query([1, 1]), InferenceMode::Frequentist);
    let program = compiled(&study);
    assert_eq!(program.route(), "SequentialContrast");
    assert_eq!(program.depth(), LoweringDepth::Complete);

    let ProblemTarget::Target(target) = program.scientific().target() else {
        panic!("the sequential contrast is world-indexed");
    };
    assert!(matches!(target.form(), TargetForm::Contrast(_)));
    assert_eq!(target.assumptions().graph(), GraphFamily::Graphless);
    assert!(!target.assumptions().records().is_empty(), "the design premises are recorded");
    assert!(program.identification_retained());
    let premises = retained_assumptions(&program).expect("the design product is retained");
    assert_eq!(&premises, target.assumptions().records());

    let statistical = program.statistical();
    let recorded = recorded_estimator(&study);
    assert!(recorded.is_some(), "the route record names its estimator");
    assert_eq!(statistical.estimator(), recorded);
    assert!(matches!(statistical.realization(), Realization::Frequentist(_)));
    let folds = statistical.folds().expect("the contrast is cross-fit");
    assert_eq!(folds.folds(), 2);
    assert_eq!(folds.unit(), ResampleUnit::Subject);
    assert!(statistical.resampling().is_none(), "the route takes no bootstrap");
    assert_eq!(statistical.models().len(), 3);
    let design = program.physical().design();
    assert!(matches!(design.source(), WidthSource::NotDesignBased(_)));
    // A route that fits no design states no rows or columns; the statistical
    // program carries the same absent shape.
    assert_eq!((design.rows(), design.columns()), (None, None));
    assert_eq!(statistical.design(), design);

    assert_eq!(compiled(&study).identity(), program.identity(), "the identity is stable");
    let bayesian = compiled(&sequential_study(
        &data,
        contrast_query([1, 1]),
        InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(DRAWS)),
    ));
    assert!(matches!(bayesian.statistical().realization(), Realization::Bayesian(_)));
    assert_eq!(bayesian.statistical().posterior().map(|plan| plan.draws()), Some(DRAWS));
    assert_ne!(bayesian.identity(), program.identity(), "inference enters the identity");
    let other_regime =
        compiled(&sequential_study(&data, contrast_query([1, 0]), InferenceMode::Frequentist));
    assert_ne!(other_regime.identity(), program.identity(), "the target enters the identity");

    let ctx = ExecutionContext::for_tests(41);
    let prepared = study.prepare(&ctx).unwrap();
    assert_eq!(prepared.plan().lowered_program_identity(), Some(program.identity()));
    let result = prepared.estimate(&data, &ctx).unwrap();
    assert_eq!(result.lowered_program_identity(), Some(program.identity()));
}

/// Columns h1, h2, y, treated, period, weight over nine units and three calendar periods.
fn history_table() -> TabularData {
    type DesignRow = (f64, [f64; 2], f64, f64, f64);
    let rows: [DesignRow; 9] = [
        (1.0, [1.0, 3.0], 10.0, 1.0, 1.0),
        (1.0, [1.0, 1.0], 4.0, 0.0, 1.0),
        (1.0, [3.0, 3.0], 6.0, 0.0, 1.0),
        (1.0, [2.8, 3.0], 9.0, 1.0, 2.0),
        (2.0, [2.0, 2.0], 5.0, 0.0, 1.0),
        (2.0, [6.0, 6.0], 7.0, 0.0, 1.0),
        (2.0, [5.0, 6.0], 12.0, 1.0, 1.5),
        (2.0, [9.0, 9.0], 20.0, 1.0, 1.0),
        (3.0, [1.0, 1.0], 3.0, 1.0, 1.0),
    ];
    let column = |f: &dyn Fn(&DesignRow) -> f64| -> Vec<f64> { rows.iter().map(f).collect() };
    let (h1, h2) = (column(&|r| r.1[0]), column(&|r| r.1[1]));
    let (y, treated) = (column(&|r| r.2), column(&|r| r.3));
    let (period, weight) = (column(&|r| r.0), column(&|r| r.4));
    TabularData::from_f64_columns([
        ("h1", h1.as_slice()),
        ("h2", h2.as_slice()),
        ("y", y.as_slice()),
        ("treated", treated.as_slice()),
        ("period", period.as_slice()),
        ("weight", weight.as_slice()),
    ])
    .unwrap()
}

fn history_query() -> HistoryMatchQuery {
    HistoryMatchQuery {
        outcome: VariableId::from_raw(2),
        treated: VariableId::from_raw(3),
        history: Arc::from([VariableId::from_raw(0), VariableId::from_raw(1)]),
        period: Some(VariableId::from_raw(4)),
        weight: Some(VariableId::from_raw(5)),
        caliper: Some(2.0),
    }
}

#[test]
fn history_match_lowers_its_matched_functional_and_retains_its_premises() {
    let data = history_table();
    let study = Study::tabular(data.clone())
        .query(CausalQuery::HistoryMatch(history_query()))
        .build()
        .unwrap();
    let program = compiled(&study);
    assert_eq!(program.route(), "HistoryMatch");
    assert_eq!(program.depth(), LoweringDepth::Complete);
    assert!(matches!(
        program.scientific().target(),
        ProblemTarget::QueryFunctional {
            kind: FunctionalKind::HistoryMatch,
            outcome: Some(outcome),
            ..
        }
            if *outcome == VariableId::from_raw(2)
    ));
    assert!(program.identification_retained());
    let premises = retained_assumptions(&program).expect("the design product is retained");
    assert!(!premises.is_empty(), "unconfoundedness given the score and overlap are recorded");
    let statistical = program.statistical();
    assert_eq!(statistical.estimator(), recorded_estimator(&study));
    assert!(matches!(statistical.realization(), Realization::Frequentist(_)));
    assert_eq!(statistical.models().len(), 2);
    assert!(statistical.folds().is_none());
    assert_eq!(compiled(&study).identity(), program.identity(), "the identity is stable");

    let ctx = ExecutionContext::for_tests(44);
    let prepared = study.prepare(&ctx).unwrap();
    assert_eq!(prepared.plan().lowered_program_identity(), Some(program.identity()));
    let result = prepared.estimate(&data, &ctx).unwrap();
    assert_eq!(result.lowered_program_identity(), Some(program.identity()));
}
