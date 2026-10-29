//! Degenerate designs and held-out intervention levels across the static effect routes.
//!
//! One structural model, `y = 3 a + 4 m + z` with `m = 2 a + noise` and `a` driven by the
//! confounder `z`, is run through the average effect, natural-direct mediation, counterfactual,
//! mean-curve and average-derivative routes. On the baseline design each recovers its closed
//! form (total effect 11, natural direct effect 3, mean unit effect 11, curve rise 11, average
//! derivative 11). A treatment column with no variation is then refused by every route instead
//! of yielding a number; an exactly duplicated confounder is refused as rank deficient by the
//! regression routes and leaves the penalized smoothers' estimates unchanged; and an
//! intervention level far outside the observed treatment range is refused by the local curve
//! and flagged extrapolative by the counterfactual.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::{CausalError, RefuteSuite, Study, StudyResult};
use antecedent_core::{
    AverageEffectQuery, CausalQuery, ContinuousDomain, CounterfactualQuery, DerivativeWeighting,
    ExecutionContext, GridSpec, Intervention, MediationContrast, MediationQuery, ResponseFunctional,
    ResponseIdentification, ResponseQuery, ResponseValue, Value, VariableId,
};
use antecedent_data::TabularData;
use antecedent_graph::{Admg, Cpdag, Dag, DenseNodeId};

fn v(raw: u32) -> VariableId {
    VariableId::from_raw(raw)
}

/// Columns `a, m, y, z` (and `z2 = 2 z + 1` when `collinear`). A constant treatment fixes
/// `a = 1`.
fn design(constant: bool, collinear: bool) -> (TabularData, Dag) {
    let n = 300_usize;
    let at = |i: usize| i as f64;
    let z: Vec<f64> = (0..n).map(|i| (at(i) * 0.37).sin()).collect();
    let a: Vec<f64> =
        (0..n).map(|i| if constant { 1.0 } else { z[i] + (at(i) * 0.71).cos() }).collect();
    let m: Vec<f64> = (0..n).map(|i| 2.0 * a[i] + (at(i) * 1.13).cos()).collect();
    let y: Vec<f64> =
        (0..n).map(|i| 3.0 * a[i] + 4.0 * m[i] + z[i] + 0.1 * (at(i) * 0.31).sin()).collect();
    let mut columns = vec![("a", a), ("m", m), ("y", y), ("z", z.clone())];
    if collinear {
        columns.push(("z2", z.iter().map(|value| 2.0 * value + 1.0).collect()));
    }
    let data =
        TabularData::from_f64_columns(columns.iter().map(|(name, c)| (*name, c.as_slice())))
            .unwrap();
    let mut dag = Dag::with_variables(u32::try_from(columns.len()).unwrap());
    let mut edges = vec![(3, 0), (3, 2), (0, 1), (1, 2), (0, 2)];
    if collinear {
        edges.extend([(4, 0), (4, 2)]);
    }
    for (from, to) in edges {
        dag.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
    }
    (data, dag)
}

fn curve(grid: &[f64]) -> CausalQuery {
    CausalQuery::Response(ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: v(2),
        treatment: ContinuousDomain::new(v(0), GridSpec::Values(grid.to_vec().into())),
    }))
}

fn counterfactual(level: f64) -> CausalQuery {
    CausalQuery::Counterfactual(
        CounterfactualQuery::new(
            v(2),
            Arc::from([Intervention::set(v(0), Value::f64(level))]),
        )
        .with_control_level(0.0),
    )
}

/// The five routes: name and query.
fn routes() -> Vec<(&'static str, CausalQuery)> {
    let mut mediation =
        MediationQuery::binary(v(0), v(2), Arc::from([v(1)]), MediationContrast::NaturalDirect);
    mediation.control = Intervention::set(v(0), Value::f64(0.0));
    mediation.active = Intervention::set(v(0), Value::f64(1.0));
    vec![
        ("average effect", CausalQuery::AverageEffect(AverageEffectQuery::binary_ate(v(0), v(2)))),
        ("mediation", CausalQuery::Mediation(mediation)),
        ("counterfactual", counterfactual(1.0)),
        ("mean curve", curve(&[-0.5, 0.0, 0.5])),
        (
            "average derivative",
            CausalQuery::Response(ResponseQuery::new(ResponseFunctional::AverageDerivative {
                outcome: v(2),
                treatment: v(0),
                weighting: DerivativeWeighting::Observed,
            })),
        ),
    ]
}

fn run(query: &CausalQuery, constant: bool, collinear: bool) -> Result<StudyResult, CausalError> {
    let (data, dag) = design(constant, collinear);
    Study::tabular(data)
        .graph(dag)
        .query(query.clone())
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .and_then(|study| study.run(&ExecutionContext::for_tests(5)))
}

/// The route's headline number: the effect, or the rise of the curve across its grid.
fn headline(name: &str, result: &StudyResult) -> f64 {
    match name {
        "counterfactual" => result.counterfactual().unwrap().mean_ite,
        "mean curve" => {
            let ResponseIdentification::PointIdentified(ResponseValue::Surface { mean, .. }) =
                &result.response().unwrap().estimate
            else {
                panic!("a point-identified surface");
            };
            mean[2] - mean[0]
        }
        _ => result.estimate().as_effect().unwrap().ate,
    }
}

#[test]
fn every_static_route_recovers_its_closed_form_effect_on_the_baseline_design() {
    for (name, query) in routes() {
        let truth = if name == "mediation" { 3.0 } else { 11.0 };
        let got = headline(name, &run(&query, false, false).unwrap());
        assert!((got - truth).abs() < 0.3, "{name}: {got} vs {truth}");
    }
}

#[test]
fn a_treatment_column_without_variation_is_refused_by_every_static_route() {
    for (name, query) in routes() {
        let error = run(&query, true, false).err().unwrap_or_else(|| {
            panic!("{name} published an estimate for a treatment column with one value")
        });
        let text = error.to_string().to_lowercase();
        assert!(
            ["rank deficient", "no mechanism candidates", "singular", "reason=arm_not_populated:"]
                .iter()
                .any(|needle| text.contains(needle)),
            "{name}: {text}"
        );
    }
}

#[test]
fn a_duplicated_confounder_is_refused_by_regression_routes_and_absorbed_by_penalized_smoothers() {
    for (name, query) in routes() {
        let baseline = headline(name, &run(&query, false, false).unwrap());
        match name {
            "mean curve" | "average derivative" => {
                let doubled = headline(name, &run(&query, false, true).unwrap());
                assert!((doubled - baseline).abs() < 0.05, "{name}: {doubled} vs {baseline}");
            }
            _ => {
                let text = run(&query, false, true)
                    .err()
                    .unwrap_or_else(|| panic!("{name} accepted an exactly duplicated confounder"))
                    .to_string();
                assert!(
                    text.contains("rank deficient") || text.contains("no mechanism candidates"),
                    "{name}: {text}"
                );
            }
        }
    }
}

#[test]
fn an_intervention_level_far_outside_the_observed_range_is_refused_or_flagged_extrapolative() {
    // The treatment ranges over about [-2, 2]. A local curve at 10 has no data under its window.
    let error = run(&curve(&[-0.5, 0.0, 10.0]), false, false).err().expect("refused");
    assert!(error.to_string().contains("too little effective weight"), "{error}");

    // The counterfactual extrapolates its fitted mechanism but says so, per unit and overall.
    let flag = |level: f64| {
        let result = run(&counterfactual(level), false, false).unwrap();
        let support = result
            .diagnostics
            .iter()
            .find(|d| d.code.as_ref() == "gcm.counterfactual.support")
            .expect("a support diagnostic");
        let text = format!("{support:?}");
        (
            text.contains("(\"extrapolative\", \"true\")"),
            text.contains("(\"per_unit_extrapolative\", \"300\")"),
        )
    };
    assert_eq!(flag(1.0), (false, false));
    assert_eq!(flag(10.0), (true, true));
}

/// The same model with every edge compelled, as a CPDAG and as an ADMG without bidirected edges.
fn class_graphs(collinear: bool) -> (Cpdag, Admg) {
    let k = if collinear { 5 } else { 4 };
    let mut edges = vec![(3, 0), (3, 2), (0, 1), (1, 2), (0, 2)];
    if collinear {
        edges.extend([(4, 0), (4, 2)]);
    }
    let mut cpdag = Cpdag::with_variables(k);
    let mut admg = Admg::with_variables(k);
    for (from, to) in edges {
        let (from, to) = (DenseNodeId::from_raw(from), DenseNodeId::from_raw(to));
        cpdag.insert_directed(from, to).unwrap();
        admg.insert_directed(from, to).unwrap();
    }
    (cpdag, admg)
}

fn run_class(class: &str, constant: bool, collinear: bool) -> Result<StudyResult, CausalError> {
    let (data, _) = design(constant, collinear);
    let (cpdag, admg) = class_graphs(collinear);
    let query = CausalQuery::AverageEffect(AverageEffectQuery::binary_ate(v(0), v(2)));
    let builder = Study::tabular(data);
    let builder = if class == "cpdag" { builder.graph(cpdag) } else { builder.graph(admg) };
    builder
        .query(query)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .and_then(|study| study.run(&ExecutionContext::for_tests(5)))
}

#[test]
fn the_equivalence_class_and_latent_variable_average_effect_routes_share_the_degeneracy_boundaries() {
    for class in ["cpdag", "admg"] {
        let effect = run_class(class, false, false).unwrap();
        let ate = effect.estimate().as_effect().unwrap().ate;
        assert!((ate - 11.0).abs() < 0.3, "{class}: baseline effect {ate}");
        for (constant, collinear, what) in [(true, false, "constant treatment"), (false, true, "duplicated confounder")] {
            let error = run_class(class, constant, collinear)
                .err()
                .unwrap_or_else(|| panic!("{class} published an effect for a {what}"));
            let text = error.to_string().to_lowercase();
            // A constant treatment leaves the contrasted arm empty; a duplicated confounder
            // leaves the design rank deficient.
            let expected: &[&str] = if constant {
                &["reason=arm_not_populated:", "has no complete-case rows"]
            } else {
                &["rank deficient", "singular", "no variation", "no mechanism"]
            };
            assert!(
                if constant {
                    expected.iter().all(|needle| text.contains(needle))
                } else {
                    expected.iter().any(|needle| text.contains(needle))
                },
                "{class} {what}: {text}"
            );
        }
    }
}
