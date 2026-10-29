// SPDX-License-Identifier: MIT OR Apache-2.0
//! The instrumental routes refuse a regime the data never visited.
//!
//! A binary treatment driven by the instrument has the known two-level contrast; asking for a
//! contrast at levels the treatment never takes would return the slope scaled by the gap to a
//! level with no rows, so it is refused with `arm_not_populated` naming the observed levels. An
//! instrument that never moves has no second arm and is refused too.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::{CausalError, IdentifierId, RefuteSuite, Study, StudyResult};
use antecedent_core::{AverageEffectQuery, ExecutionContext, VariableId};
use antecedent_data::{TableView, TabularData};
use antecedent_estimate::{AnalyticSeKind, TwoStageLeastSquares, WaldIv};
use antecedent_graph::Dag;

fn run(
    t: &[f64],
    y: &[f64],
    z: &[f64],
    levels: (f64, f64),
    wald: bool,
) -> Result<StudyResult, CausalError> {
    let data = TabularData::from_f64_columns([("t", t), ("y", y), ("z", z)]).unwrap();
    let estimator = if wald {
        antecedent::EstimatorSpec::from(WaldIv::new().with_se_kind(AnalyticSeKind::Hc1))
    } else {
        antecedent::EstimatorSpec::from(
            TwoStageLeastSquares::new().with_se_kind(AnalyticSeKind::Hc1),
        )
    };
    Study::tabular(data.clone())
        .query(AverageEffectQuery::with_levels(
            VariableId::from_raw(0),
            VariableId::from_raw(1),
            levels.0,
            levels.1,
        ))
        .identifier(IdentifierId::Iv)
        .estimator(estimator)
        .refute(RefuteSuite::None)
        .graph(Dag::from_named_edges(data.schema(), &[("z", "t"), ("t", "y")]).unwrap())
        .build()
        .and_then(|study| study.run(&ExecutionContext::for_tests(5)))
}

fn binary_design() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = 800;
    let z: Vec<f64> = (0..n).map(|i| (i % 2) as f64).collect();
    let u: Vec<f64> = (0..n).map(|i| ((i * 37 % 101) as f64 - 50.0) / 50.0).collect();
    let t: Vec<f64> =
        (0..n).map(|i| if 0.8 * z[i] + 0.3 * u[i] + 0.1 > 0.5 { 1.0 } else { 0.0 }).collect();
    let y: Vec<f64> = (0..n).map(|i| 2.0 * t[i] + u[i]).collect();
    (t, y, z)
}

#[test]
fn cells_00_iv_refuses_contrast_levels_the_discrete_treatment_never_takes() {
    let (t, y, z) = binary_design();
    for wald in [false, true] {
        // The observed pair recovers the structural effect 2 (the route works).
        let ok = run(&t, &y, &z, (0.0, 1.0), wald).unwrap();
        let ate = ok.estimate().as_effect().unwrap().ate;
        assert!((ate - 2.0).abs() < 0.1, "wald {wald}: {ate}");
        for levels in [(0.0, 5.0), (2.0, 3.0), (-1.0, 1.0)] {
            let error = run(&t, &y, &z, levels, wald).err().unwrap_or_else(|| {
                panic!("wald {wald} {levels:?}: scaled a slope to an unvisited level")
            });
            assert_eq!(
                error.reason_code(),
                Some("arm_not_populated"),
                "wald {wald} {levels:?}: {error}"
            );
            let text = error.to_string();
            assert!(text.contains("observed treatment levels: 0 ("), "{text}");
            assert!(text.contains("Next attempt"), "{text}");
        }
    }
}

#[test]
fn cells_00_iv_refuses_an_instrument_that_never_moves() {
    let (t, y, _) = binary_design();
    let constant = vec![1.0; t.len()];
    for wald in [false, true] {
        let error = run(&t, &y, &constant, (0.0, 1.0), wald)
            .err()
            .unwrap_or_else(|| panic!("wald {wald}: a constant instrument produced an estimate"));
        let text = error.to_string();
        assert!(
            text.contains("rank deficient") || text.contains("requires both instrument arms"),
            "wald {wald}: {text}"
        );
    }
}
