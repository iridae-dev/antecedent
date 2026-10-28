//! A contrast of two levels of a discrete treatment, through the regression estimators.
//!
//! The outcome is `y = f(t) + z / 2` with `f = (0, 1, 4)` at the treatment levels `0, 1, 2` and
//! a binary covariate `z` balanced within every level, so the contrast of level 1 against level
//! 0 is exactly 1 and of level 2 against level 0 exactly 4. A regression that enters the
//! treatment as one numeric column would read the three levels as a dose (slope 2, so 2 for "1
//! against 0"); on the adjusted graph and on an equivalence class it refuses the column with
//! `treatment_level_undeclared`, naming the observed levels, unless the restriction to the rows
//! at the two levels is declared, in which case the contrast is exact and the excluded rows are
//! reported.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::{CausalError, RefuteSuite, Study, StudyResult};
use antecedent_core::{AverageEffectQuery, ExecutionContext, VariableId};
use antecedent_data::TabularData;
use antecedent_estimate::{GlmAdjustmentAte, LinearAdjustmentAte, OtherLevels};
use antecedent_graph::{Cpdag, Dag, DenseNodeId};
use antecedent_stats::GlmFamily;

const N: usize = 120;
const LISTING: &str = "observed treatment levels: 0 (40 rows), 1 (40 rows), 2 (40 rows)";

fn table() -> TabularData {
    let t: Vec<f64> = (0..N).map(|i| (i % 3) as f64).collect();
    let z: Vec<f64> = (0..N).map(|i| ((i / 3) % 2) as f64).collect();
    let y: Vec<f64> = (0..N).map(|i| [0.0, 1.0, 4.0][i % 3] + 0.5 * z[i]).collect();
    TabularData::from_f64_columns([("t", t.as_slice()), ("y", y.as_slice()), ("z", z.as_slice())])
        .unwrap()
}

fn edges() -> [(u32, u32); 3] {
    [(2, 0), (2, 1), (0, 1)]
}

fn dag() -> Dag {
    let mut graph = Dag::with_variables(3);
    for (from, to) in edges() {
        graph.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
    }
    graph
}

fn cpdag() -> Cpdag {
    let mut graph = Cpdag::with_variables(3);
    for (from, to) in edges() {
        graph.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
    }
    graph
}

fn contrast(class: &str, estimator: impl Into<antecedent::EstimatorSpec>, active: f64) -> Result<StudyResult, CausalError> {
    let query = AverageEffectQuery::with_levels(VariableId::from_raw(0), VariableId::from_raw(1), 0.0, active);
    let builder = Study::tabular(table());
    let builder = if class == "cpdag" { builder.graph(cpdag()) } else { builder.graph(dag()) };
    builder
        .query(query)
        .estimator(estimator)
        .refute(RefuteSuite::None)
        .build()
        .and_then(|study| study.run(&ExecutionContext::for_tests(5)))
}

fn estimators(
    other_levels: Option<OtherLevels>,
) -> Vec<(&'static str, antecedent::EstimatorSpec)> {
    let linear = LinearAdjustmentAte::new().with_bootstrap_replicates(0);
    let glm = GlmAdjustmentAte::new().with_family(GlmFamily::GaussianIdentity).with_bootstrap_replicates(0);
    match other_levels {
        None => vec![("linear", linear.into()), ("gaussian glm", glm.into())],
        Some(policy) => vec![
            ("linear", linear.with_other_levels(policy).into()),
            ("gaussian glm", glm.with_other_levels(policy).into()),
        ],
    }
}

#[test]
fn a_three_level_treatment_is_refused_naming_its_levels_on_the_adjusted_and_equivalence_class_routes() {
    for class in ["dag", "cpdag"] {
        for (name, estimator) in estimators(None) {
            let error = contrast(class, estimator, 1.0)
                .err()
                .unwrap_or_else(|| panic!("{class} {name}: a three-level treatment read as a dose"));
            assert_eq!(error.reason_code(), Some("treatment_level_undeclared"), "{class} {name}: {error}");
            let text = error.to_string();
            assert!(text.contains(LISTING), "{class} {name}: {text}");
            assert!(text.contains("Next attempt"), "{class} {name}: {text}");
        }
    }
}

#[test]
fn the_declared_restriction_gives_the_contrast_of_the_two_levels_on_every_route() {
    for class in ["dag", "cpdag"] {
        for (active, truth) in [(1.0, 1.0), (2.0, 4.0)] {
            for (name, estimator) in estimators(Some(OtherLevels::ExcludeUndeclared)) {
                let result = contrast(class, estimator, active)
                    .unwrap_or_else(|error| panic!("{class} {name} {active}: {error}"));
                let effect = result.estimate().as_effect().unwrap();
                assert!(
                    (effect.ate - truth).abs() < 1e-9,
                    "{class} {name}: {} vs {truth}",
                    effect.ate
                );
                let levels = effect
                    .diagnostics
                    .treatment_levels
                    .unwrap_or_else(|| panic!("{class} {name}: the exclusion is not reported"));
                assert_eq!(levels.excluded_rows, 40, "{class} {name}");
            }
        }
    }
}
