//! The static effect and response routes on a latent-variable graph and on an equivalence class.
//!
//! One structural model, `y = 3 a + 4 m + z` with `m = 2 a + noise` and `a` driven by the
//! confounder `z`, is given to each route as an ADMG with no bidirected edge and as a CPDAG
//! whose edges are all compelled. Every route that runs recovers its closed form on both
//! classes: total effect 11 for the average, conditional, mean-curve and average-derivative
//! routes, the mean unit effect 11 for the counterfactual, and for the intervention response the
//! mean of the structural equations written out row by row. The routes that do not run are
//! refused with their own typed detail rather than approximated. Observation weights on the
//! table are refused, estimator-declared weights equal the same fit on rows repeated by integer
//! weights, and a missing confounder follows the default complete-case policy, equal to the fit on
//! the rows that remain. On both graph classes a declared cluster or Newey-West error is the
//! closed-form sandwich of the regression scores; when the class holds two completions with
//! different adjustment sets, the error is that sandwich over their frozen-weight mixture of
//! influence values.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::{CausalError, RefuteSuite, Study, StudyResult};
use antecedent_core::{
    AverageEffectQuery, CausalQuery, ConditionalEffectQuery, ContinuousDomain, CounterfactualQuery,
    DerivativeWeighting, ExecutionContext, GridSpec, Intervention, MediationContrast,
    MediationQuery, ResponseFunctional, ResponseIdentification, ResponseQuery, ResponseValue,
    Value, VariableId,
};
use antecedent_data::{MissingDeclaration, MissingPolicy, TableView, TabularData};
use antecedent_estimate::{AnalyticSeKind, LinearAdjustmentAte, SamplingWeights};
use antecedent_graph::{Admg, Cpdag, Dag, DenseNodeId};

fn v(raw: u32) -> VariableId {
    VariableId::from_raw(raw)
}

const N: usize = 300;
const CLASSES: [&str; 2] = ["admg", "cpdag"];

/// Columns `a, m, y, z`, written as the structural equations read. A constant treatment fixes
/// `a = 1`; `collinear` adds `z2 = 2 z + 1`, an exact copy of the confounder up to scale.
fn design(constant: bool, collinear: bool) -> Vec<(&'static str, Vec<f64>)> {
    let at = |i: usize| i as f64;
    let z: Vec<f64> = (0..N).map(|i| (at(i) * 0.37).sin()).collect();
    let a: Vec<f64> =
        (0..N).map(|i| if constant { 1.0 } else { z[i] + (at(i) * 0.71).cos() }).collect();
    let m: Vec<f64> = (0..N).map(|i| 2.0 * a[i] + (at(i) * 1.13).cos()).collect();
    let y: Vec<f64> =
        (0..N).map(|i| 3.0 * a[i] + 4.0 * m[i] + z[i] + 0.1 * (at(i) * 0.31).sin()).collect();
    let mut columns = vec![("a", a), ("m", m), ("y", y), ("z", z.clone())];
    if collinear {
        columns.push(("z2", z.iter().map(|value| 2.0 * value + 1.0).collect()));
    }
    columns
}

fn columns() -> Vec<(&'static str, Vec<f64>)> {
    design(false, false)
}

/// The directed edges over `k` variables: the model's five, and the copy's two when `k` is five.
fn edges(k: u32) -> Vec<(u32, u32)> {
    let mut edges = vec![(3, 0), (3, 2), (0, 1), (1, 2), (0, 2)];
    if k == 5 {
        edges.extend([(4, 0), (4, 2)]);
    }
    edges
}

fn table(columns: &[(&str, Vec<f64>)]) -> TabularData {
    TabularData::from_f64_columns(columns.iter().map(|(name, c)| (*name, c.as_slice()))).unwrap()
}

fn admg(k: u32) -> Admg {
    let mut graph = Admg::with_variables(k);
    for (from, to) in edges(k) {
        graph.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
    }
    graph
}

fn cpdag(k: u32) -> Cpdag {
    let mut graph = Cpdag::with_variables(k);
    for (from, to) in edges(k) {
        graph.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
    }
    graph
}

fn dag(k: u32) -> Dag {
    let mut graph = Dag::with_variables(k);
    for (from, to) in edges(k) {
        graph.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
    }
    graph
}

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(5)
}

fn run(
    class: &str,
    data: TabularData,
    query: CausalQuery,
    estimator: Option<LinearAdjustmentAte>,
    missing: Option<MissingDeclaration>,
) -> Result<StudyResult, CausalError> {
    let k = u32::try_from(data.schema().len()).unwrap();
    let builder = Study::tabular(data);
    let builder = match class {
        "cpdag" => builder.graph(cpdag(k)),
        "dag" => builder.graph(dag(k)),
        _ => builder.graph(admg(k)),
    };
    let builder = match estimator {
        Some(estimator) => builder.estimator(estimator),
        None => builder.refute(RefuteSuite::None).bootstrap_replicates(0),
    };
    let builder = match missing {
        Some(declaration) => builder.missing(declaration),
        None => builder,
    };
    builder.query(query).build().and_then(|study| study.run(&ctx()))
}

fn plain(class: &str, query: &CausalQuery) -> Result<StudyResult, CausalError> {
    run(class, table(&columns()), query.clone(), None, None)
}

fn average_effect() -> CausalQuery {
    CausalQuery::AverageEffect(AverageEffectQuery::binary_ate(v(0), v(2)))
}

fn mean_curve() -> CausalQuery {
    CausalQuery::Response(ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: v(2),
        treatment: ContinuousDomain::new(v(0), GridSpec::Values(vec![-0.5, 0.0, 0.5].into())),
    }))
}

fn rise(result: &StudyResult) -> f64 {
    let ResponseIdentification::PointIdentified(ResponseValue::Surface { mean, .. }) =
        &result.response().unwrap().estimate
    else {
        panic!("a point-identified surface");
    };
    mean[mean.len() - 1] - mean[0]
}

fn scalar(result: &StudyResult) -> f64 {
    match &result.response().unwrap().estimate {
        ResponseIdentification::PointIdentified(ResponseValue::Scalar(value)) => *value,
        other => panic!("a point-identified scalar, got {other:?}"),
    }
}

fn mediation() -> CausalQuery {
    let mut query =
        MediationQuery::binary(v(0), v(2), Arc::from([v(1)]), MediationContrast::NaturalDirect);
    query.control = Intervention::set(v(0), Value::f64(0.0));
    query.active = Intervention::set(v(0), Value::f64(1.0));
    CausalQuery::Mediation(query)
}

fn counterfactual() -> CausalQuery {
    CausalQuery::Counterfactual(
        CounterfactualQuery::new(v(2), Arc::from([Intervention::set(v(0), Value::f64(1.0))]))
            .with_control_level(0.0),
    )
}

#[test]
fn the_effect_and_response_routes_recover_their_closed_forms_on_both_graph_classes() {
    let conditional = CausalQuery::ConditionalEffect(
        ConditionalEffectQuery::try_new(
            AverageEffectQuery::binary_ate(v(0), v(2)).with_effect_modifiers(vec![v(3)]),
        )
        .unwrap(),
    );
    let derivative = CausalQuery::Response(ResponseQuery::new(ResponseFunctional::AverageDerivative {
        outcome: v(2),
        treatment: v(0),
        weighting: DerivativeWeighting::Observed,
    }));
    // E[y | do(a = 0.25)] is the mean of the structural equation with `a` set and the
    // exogenous terms of every row kept.
    let at = |i: usize| i as f64;
    let truth: f64 = (0..N)
        .map(|i| {
            let z = (at(i) * 0.37).sin();
            3.0 * 0.25 + 4.0 * (2.0 * 0.25 + (at(i) * 1.13).cos()) + z + 0.1 * (at(i) * 0.31).sin()
        })
        .sum::<f64>()
        / N as f64;
    let intervention = CausalQuery::Response(ResponseQuery::new(
        ResponseFunctional::InterventionResponse {
            outcome: v(2),
            interventions: Arc::from([Intervention::set(v(0), Value::f64(0.25))]),
        },
    ));
    for class in CLASSES {
        for (name, query) in [("average effect", average_effect()), ("conditional effect", conditional.clone())] {
            let ate = plain(class, &query).unwrap().estimate().as_effect().unwrap().ate;
            assert!((ate - 11.0).abs() < 0.3, "{class} {name}: {ate}");
        }
        let curve = rise(&plain(class, &mean_curve()).unwrap());
        assert!((curve - 11.0).abs() < 0.3, "{class} mean curve rise {curve}");
        let slope = scalar(&plain(class, &derivative).unwrap());
        assert!((slope - 11.0).abs() < 0.3, "{class} average derivative {slope}");
        let level = scalar(&plain(class, &intervention).unwrap());
        assert!((level - truth).abs() < 0.25, "{class} intervention response {level} vs {truth}");
    }
    // The conditional effect also runs on the plain DAG of the same model and agrees to the
    // ADMG: a graph with no bidirected edge adds nothing.
    let on_dag = plain("dag", &conditional).unwrap().estimate().as_effect().unwrap().ate;
    let on_admg = plain("admg", &conditional).unwrap().estimate().as_effect().unwrap().ate;
    assert!((on_dag - 11.0).abs() < 0.3 && (on_dag - on_admg).abs() < 1e-9, "{on_dag} {on_admg}");
    // The counterfactual runs on the equivalence class and recovers the mean unit effect 11.
    let (on_admg, on_cpdag) = (
        scalar(&plain("admg", &intervention).unwrap()),
        scalar(&plain("cpdag", &intervention).unwrap()),
    );
    assert!((on_admg - on_cpdag).abs() < 1e-9, "{on_admg} vs {on_cpdag}");
    let unit_effect = plain("cpdag", &counterfactual()).unwrap().counterfactual().unwrap().mean_ite;
    assert!((unit_effect - 11.0).abs() < 0.3, "cpdag counterfactual {unit_effect}");
}

#[test]
fn the_routes_with_no_latent_variable_or_class_construction_are_refused_with_their_own_detail() {
    // The latent-variable counterfactual is a discrete identification with at most four levels.
    let error = plain("admg", &counterfactual()).err().expect("a refusal");
    assert_eq!(error.reason_code(), Some("route_not_supported"), "{error}");
    assert!(error.to_string().contains("counterfactual_id.bounds_exceeded"), "{error}");
    // Natural-direct mediation over the graph classes is a discrete functional.
    for class in CLASSES {
        let error = plain(class, &mediation()).err().expect("a refusal");
        assert!(error.to_string().contains("discrete level cap"), "{class}: {error}");
    }
}

fn weights() -> Vec<f64> {
    (0..N).map(|i| 1.0 + ((i * 7) % 4) as f64).collect()
}

fn replicated(weights: &[f64]) -> TabularData {
    let expanded: Vec<(&str, Vec<f64>)> = columns()
        .iter()
        .map(|(name, c)| {
            let rows =
                c.iter().zip(weights).flat_map(|(&x, &w)| std::iter::repeat_n(x, w as usize));
            (*name, rows.collect())
        })
        .collect();
    table(&expanded)
}

fn linear(weights: Option<Vec<f64>>) -> LinearAdjustmentAte {
    let estimator =
        LinearAdjustmentAte::new().with_bootstrap_replicates(0).with_se_kind(AnalyticSeKind::Hc1);
    match weights {
        Some(weights) => estimator.with_sampling_weights(SamplingWeights::new(weights).unwrap()),
        None => estimator,
    }
}

#[test]
fn table_weights_are_refused_and_declared_weights_equal_row_replication_on_both_graph_classes() {
    for class in CLASSES {
        for w in [weights(), vec![1.0; N]] {
            let weighted = table(&columns()).with_weights(Arc::from(w)).unwrap();
            let error = run(class, weighted, average_effect(), None, None)
                .err()
                .unwrap_or_else(|| panic!("{class}: published an estimate for weighted rows"));
            let text = error.to_string();
            assert!(text.starts_with("reason=option_not_applicable"), "{class}: {text}");
            assert!(text.contains("sampling_weights.route_unsupported"), "{class}: {text}");
        }
        let honoured =
            run(class, table(&columns()), average_effect(), Some(linear(Some(weights()))), None)
                .unwrap();
        let copies =
            run(class, replicated(&weights()), average_effect(), Some(linear(None)), None).unwrap();
        let unweighted =
            run(class, table(&columns()), average_effect(), Some(linear(None)), None).unwrap();
        let (got, want, plain) = (
            honoured.estimate().as_effect().unwrap().ate,
            copies.estimate().as_effect().unwrap().ate,
            unweighted.estimate().as_effect().unwrap().ate,
        );
        assert!((got - want).abs() < 1e-9, "{class}: weighted {got} vs replicated {want}");
        assert!((got - plain).abs() > 1e-4, "{class}: the weights changed nothing ({got})");
    }
}

#[test]
fn a_missing_confounder_defaults_to_complete_case_and_equals_the_fit_on_the_remaining_rows() {
    let holes: Vec<usize> = (0..N).filter(|i| i % 10 == 3).collect();
    let mut holey = columns();
    for &row in &holes {
        holey[3].1[row] = f64::NAN;
    }
    let kept: Vec<(&str, Vec<f64>)> = columns()
        .into_iter()
        .map(|(name, c)| {
            (name, c.into_iter().enumerate().filter(|(i, _)| i % 10 != 3).map(|(_, x)| x).collect())
        })
        .collect();
    let remaining_ate = |class: &str| {
        run(class, table(&kept), average_effect(), None, None)
            .unwrap()
            .estimate()
            .as_effect()
            .unwrap()
            .ate
    };
    for class in CLASSES {
        let declared = run(
            class,
            table(&holey),
            average_effect(),
            None,
            Some(MissingDeclaration::new(MissingPolicy::CompleteCase)),
        )
        .unwrap();
        let default = run(class, table(&holey), average_effect(), None, None).unwrap();
        let want = remaining_ate(class);
        for (label, result) in [("declared", &declared), ("default", &default)] {
            let got = result.estimate().as_effect().unwrap().ate;
            assert!((got - want).abs() < 1e-9, "{class} {label}: {got} vs remaining {want}");
            let report = result.missing_data().as_ref().expect("a missing-data report");
            assert_eq!(report.policy(), MissingPolicy::CompleteCase, "{class} {label}");
            assert_eq!(report.accounting.rows_total, N, "{class} {label}");
            assert_eq!(report.accounting.rows_kept, N - holes.len(), "{class} {label}");
            let dropped: Vec<u64> = holes.iter().map(|&row| row as u64).collect();
            assert_eq!(report.accounting.dropped_rows.as_ref(), dropped.as_slice(), "{class}");
        }
        assert!(default.missing_data().as_ref().is_some_and(|r| !r.declared), "{class}");
        assert!(declared.missing_data().as_ref().is_some_and(|r| r.declared), "{class}");
        // An explicit error policy is the refusal.
        let error = run(
            class,
            table(&holey),
            average_effect(),
            None,
            Some(MissingDeclaration::new(MissingPolicy::ErrorOnMissing)),
        )
        .err()
        .unwrap_or_else(|| panic!("{class}: averaged over a missing confounder under `error`"));
        assert_eq!(error.reason_code(), Some("missing_policy_required"), "{class}: {error}");
    }
}

/// The conditional effect on the modifier `z` and the intervention response at `a = 0.25`.
fn remaining_routes() -> Vec<(&'static str, CausalQuery)> {
    vec![
        (
            "conditional effect",
            CausalQuery::ConditionalEffect(
                ConditionalEffectQuery::try_new(
                    AverageEffectQuery::binary_ate(v(0), v(2)).with_effect_modifiers(vec![v(3)]),
                )
                .unwrap(),
            ),
        ),
        (
            "intervention response",
            CausalQuery::Response(ResponseQuery::new(ResponseFunctional::InterventionResponse {
                outcome: v(2),
                interventions: Arc::from([Intervention::set(v(0), Value::f64(0.25))]),
            })),
        ),
    ]
}

fn text_of(result: Result<StudyResult, CausalError>, what: &str) -> String {
    result.err().unwrap_or_else(|| panic!("{what}: published an estimate")).to_string()
}

#[test]
fn the_conditional_effect_and_the_intervention_response_refuse_degenerate_designs_on_every_graph() {
    for class in ["dag", "admg", "cpdag"] {
        for (name, query) in remaining_routes() {
            let what = format!("{class} {name}");
            // The baseline design runs, so each refusal below is of the design alone.
            assert!(
                run(class, table(&design(false, false)), query.clone(), None, None).is_ok(),
                "{what}: baseline"
            );
            let constant = text_of(
                run(class, table(&design(true, false)), query.clone(), None, None),
                &what,
            );
            let copied = text_of(
                run(class, table(&design(false, true)), query.clone(), None, None),
                &what,
            );
            if name == "conditional effect" {
                // The constant treatment never visits the control level, so the arm is refused
                // before any design is built; the copy still loses a column to the rank check.
                assert!(constant.contains("arm_not_populated"), "{what}: {constant}");
                assert!(constant.contains("the treatment takes the levels 1"), "{what}: {constant}");
                assert!(copied.contains("rank deficient: rank=4 ncols=5"), "{what}: {copied}");
            } else {
                // The penalized smoother has no invertible design.
                assert!(constant.contains("singular"), "{what}: {constant}");
                assert!(copied.contains("invertible penalized design"), "{what}: {copied}");
            }
        }
    }
}

fn derivative() -> CausalQuery {
    CausalQuery::Response(ResponseQuery::new(ResponseFunctional::AverageDerivative {
        outcome: v(2),
        treatment: v(0),
        weighting: DerivativeWeighting::Observed,
    }))
}

fn headline(result: &StudyResult) -> f64 {
    if result.response().is_some() { scalar(result) } else { result.estimate().as_effect().unwrap().ate }
}

#[test]
fn the_response_routes_default_to_complete_case_and_equal_the_fit_on_the_remaining_rows() {
    let holes: Vec<usize> = (0..N).filter(|i| i % 10 == 3).collect();
    let mut holey = columns();
    for &row in &holes {
        holey[3].1[row] = f64::NAN;
    }
    let kept: Vec<(&str, Vec<f64>)> = columns()
        .into_iter()
        .map(|(name, c)| {
            (name, c.into_iter().enumerate().filter(|(i, _)| i % 10 != 3).map(|(_, x)| x).collect())
        })
        .collect();
    let routes = [
        ("average derivative", derivative()),
        ("intervention response", remaining_routes().remove(1).1),
    ];
    for (name, query) in routes {
        let remaining = headline(&run("dag", table(&kept), query.clone(), None, None).unwrap());
        let declared = run(
            "dag",
            table(&holey),
            query.clone(),
            None,
            Some(MissingDeclaration::new(MissingPolicy::CompleteCase)),
        )
        .unwrap();
        let default = run("dag", table(&holey), query.clone(), None, None).unwrap();
        for (label, result) in [("declared", &declared), ("default", &default)] {
            let got = headline(result);
            assert!((got - remaining).abs() < 1e-9, "{name} {label}: {got} vs remaining {remaining}");
            let report = result.missing_data().as_ref().expect("a missing-data report");
            assert_eq!(report.accounting.rows_kept, N - holes.len(), "{name} {label}");
        }
        let error = run(
            "dag",
            table(&holey),
            query.clone(),
            None,
            Some(MissingDeclaration::new(MissingPolicy::ErrorOnMissing)),
        )
        .err()
        .unwrap_or_else(|| panic!("{name}: averaged over a missing confounder under `error`"));
        assert_eq!(error.reason_code(), Some("missing_policy_required"), "{name}: {error}");
        // Multiple imputation completes every table and pools one finite answer.
        let imputed = run(
            "dag",
            table(&holey),
            query,
            None,
            Some(MissingDeclaration::new(MissingPolicy::MultipleImputation(
                antecedent_data::ImputationSpec::new(),
            ))),
        )
        .unwrap_or_else(|error| panic!("{name}: imputation: {error}"));
        assert!(headline(&imputed).is_finite(), "{name}");
    }
}

// ---------------------------------------------------------------------------------------------
// Declared dependence on the average effect of the graph classes
// ---------------------------------------------------------------------------------------------

fn solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Vec<f64> {
    let k = b.len();
    for col in 0..k {
        let pivot = (col..k).max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs())).unwrap();
        a.swap(col, pivot);
        b.swap(col, pivot);
        for row in col + 1..k {
            let f = a[row][col] / a[col][col];
            for c in col..k {
                a[row][c] -= f * a[col][c];
            }
            b[row] -= f * b[col];
        }
    }
    let mut x = vec![0.0; k];
    for row in (0..k).rev() {
        let tail: f64 = (row + 1..k).map(|c| a[row][c] * x[c]).sum();
        x[row] = (b[row] - tail) / a[row][row];
    }
    x
}

/// The coefficient of `a` in `y ~ 1 + a + z` and its sandwich standard error, from the
/// definition: scores `x_i e_i`, summed within `groups` (CR1) or Bartlett-weighted over `lag`
/// (Newey-West, no small-sample factor).
fn sandwich(groups: Option<&[u32]>, lag: Option<usize>) -> (f64, f64) {
    let c = columns();
    let (a, y, z) = (&c[0].1, &c[2].1, &c[3].1);
    let x: Vec<[f64; 3]> = (0..N).map(|i| [1.0, a[i], z[i]]).collect();
    let mut xtx = vec![vec![0.0; 3]; 3];
    let mut xty = vec![0.0; 3];
    for i in 0..N {
        for p in 0..3 {
            xty[p] += x[i][p] * y[i];
            for q in 0..3 {
                xtx[p][q] += x[i][p] * x[i][q];
            }
        }
    }
    let beta = solve(xtx.clone(), xty);
    let scores: Vec<[f64; 3]> = (0..N)
        .map(|i| {
            let e = y[i] - (0..3).map(|k| x[i][k] * beta[k]).sum::<f64>();
            [x[i][0] * e, x[i][1] * e, x[i][2] * e]
        })
        .collect();
    let mut omega = [[0.0_f64; 3]; 3];
    if let Some(groups) = groups {
        let mut cells = std::collections::BTreeMap::<u32, [f64; 3]>::new();
        for (s, g) in scores.iter().zip(groups) {
            let cell = cells.entry(*g).or_insert([0.0; 3]);
            for k in 0..3 {
                cell[k] += s[k];
            }
        }
        for cell in cells.values() {
            for p in 0..3 {
                for q in 0..3 {
                    omega[p][q] += cell[p] * cell[q];
                }
            }
        }
        let g = cells.len() as f64;
        let factor = g / (g - 1.0) * (N as f64 - 1.0) / (N as f64 - 3.0);
        for row in &mut omega {
            for value in row {
                *value *= factor;
            }
        }
    } else if let Some(lag) = lag {
        for l in 0..=lag {
            let weight = if l == 0 { 1.0 } else { 1.0 - l as f64 / (lag as f64 + 1.0) };
            for i in l..N {
                for p in 0..3 {
                    for q in 0..3 {
                        let term = weight * scores[i][p] * scores[i - l][q];
                        omega[p][q] += term;
                        if l > 0 {
                            omega[q][p] += term;
                        }
                    }
                }
            }
        }
    }
    let inverse: Vec<Vec<f64>> = (0..3)
        .map(|c| {
            let mut e = vec![0.0; 3];
            e[c] = 1.0;
            solve(xtx.clone(), e)
        })
        .collect();
    let variance: f64 = (0..3)
        .map(|p| (0..3).map(|q| inverse[1][p] * omega[p][q] * inverse[q][1]).sum::<f64>())
        .sum();
    (beta[1], variance.sqrt())
}

#[test]
fn declared_cluster_and_newey_west_errors_are_the_closed_form_sandwich_on_both_graph_classes() {
    let clusters: Vec<u32> = (0..N).map(|i| (i / 10) as u32).collect();
    let fit = |class: &str, estimator: LinearAdjustmentAte| {
        let result = run(class, table(&columns()), average_effect(), Some(estimator), None).unwrap();
        let effect = result.estimate().as_effect().unwrap();
        (effect.ate, effect.se_analytic())
    };
    let base = || LinearAdjustmentAte::new().with_bootstrap_replicates(0);
    let (_, independent) = fit("admg", base().with_se_kind(AnalyticSeKind::Hc1));
    // The equivalence-class (CPDAG) envelope has one completion here, whose error is the
    // declared sandwich exactly as on the latent-variable graph.
    for class in CLASSES {
        let (beta, se) = sandwich(Some(&clusters), None);
        assert!((independent - se).abs() > 1e-3 * se, "the design must make clustering matter");
        let (ate, got) = fit(
            class,
            base().with_se_kind(AnalyticSeKind::Cluster).with_cluster_ids(clusters.clone()),
        );
        assert!((ate - beta).abs() < 1e-9, "{class}: coefficient {ate} vs {beta}");
        assert!((got - se).abs() < 1e-9 * (1.0 + se), "{class}: cluster sandwich {got} vs {se}");
        for lag in [1_usize, 4] {
            let (beta, se) = sandwich(None, Some(lag));
            let (ate, got) = fit(class, base().with_se_kind(AnalyticSeKind::NeweyWest { lag }));
            assert!((ate - beta).abs() < 1e-9, "{class}: coefficient {ate} vs {beta}");
            assert!(
                (got - se).abs() < 1e-9 * (1.0 + se),
                "{class}: newey-west lag {lag}: {got} vs {se}"
            );
        }
    }
}


/// Influence values `n [(X'X)^-1 x_i e_i]_a` of the coefficient of `a` in the regression of `y`
/// on an intercept, `a` and the `extra` columns.
fn coefficient_influence(extra: &[&[f64]]) -> Vec<f64> {
    let c = columns();
    let (a, y) = (&c[0].1, &c[2].1);
    let k = 2 + extra.len();
    let x: Vec<Vec<f64>> = (0..N)
        .map(|i| {
            let mut row = vec![1.0, a[i]];
            row.extend(extra.iter().map(|column| column[i]));
            row
        })
        .collect();
    let mut xtx = vec![vec![0.0; k]; k];
    let mut xty = vec![0.0; k];
    for i in 0..N {
        for p in 0..k {
            xty[p] += x[i][p] * y[i];
            for q in 0..k {
                xtx[p][q] += x[i][p] * x[i][q];
            }
        }
    }
    let beta = solve(xtx.clone(), xty);
    let bread: Vec<f64> = {
        let mut unit = vec![0.0; k];
        unit[1] = 1.0;
        solve(xtx, unit)
    };
    (0..N)
        .map(|i| {
            let e = y[i] - (0..k).map(|p| x[i][p] * beta[p]).sum::<f64>();
            N as f64 * (0..k).map(|p| bread[p] * x[i][p] * e).sum::<f64>()
        })
        .collect()
}

/// `a` and `z` joined by an undirected edge, the rest of the model compelled: the class holds the
/// completions `z -> a` (adjust for `z`) and `a -> z` (no adjustment).
fn two_completion_cpdag() -> Cpdag {
    let mut graph = Cpdag::with_variables(4);
    for (from, to) in [(3, 2), (0, 1), (1, 2), (0, 2)] {
        graph.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
    }
    graph.insert_undirected(DenseNodeId::from_raw(3), DenseNodeId::from_raw(0)).unwrap();
    graph
}

#[test]
fn two_completions_publish_the_declared_sandwich_of_their_mixed_influence_values() {
    let clusters: Vec<u32> = (0..N).map(|i| (i / 10) as u32).collect();
    let c = columns();
    let z = c[3].1.as_slice();
    let mixed: Vec<f64> = coefficient_influence(&[z])
        .into_iter()
        .zip(coefficient_influence(&[]))
        .map(|(adjusted, unadjusted)| 0.5 * (adjusted + unadjusted))
        .collect();
    let mean = mixed.iter().sum::<f64>() / N as f64;
    let n = N as f64;
    let cluster_sums = {
        let mut sums = std::collections::BTreeMap::<u32, f64>::new();
        for (value, group) in mixed.iter().zip(&clusters) {
            *sums.entry(*group).or_default() += value - mean;
        }
        sums
    };
    let g = cluster_sums.len() as f64;
    let cluster_oracle =
        (g / (g - 1.0) * cluster_sums.values().map(|s| s * s).sum::<f64>()).sqrt() / n;
    let lag = 4_usize;
    let long_run = |l: usize| -> f64 {
        (l..N).map(|i| (mixed[i] - mean) * (mixed[i - l] - mean)).sum::<f64>()
    };
    let bartlett = long_run(0)
        + 2.0 * (1..=lag).map(|l| (1.0 - l as f64 / (lag as f64 + 1.0)) * long_run(l)).sum::<f64>();
    let newey_west_oracle = bartlett.max(0.0).sqrt() / n;
    let hc1_oracle = (mixed.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n * (n - 1.0))).sqrt();

    let fit = |estimator: LinearAdjustmentAte| {
        let result = Study::tabular(table(&columns()))
            .graph(two_completion_cpdag())
            .estimator(estimator)
            .query(average_effect())
            .build()
            .and_then(|study| study.run(&ctx()))
            .unwrap();
        result.estimate().as_effect().unwrap().se_analytic()
    };
    let base = || LinearAdjustmentAte::new().with_bootstrap_replicates(0);
    let independent = fit(base().with_se_kind(AnalyticSeKind::Hc1));
    assert!((independent - hc1_oracle).abs() < 1e-9 * (1.0 + hc1_oracle), "{independent} vs {hc1_oracle}");
    let clustered =
        fit(base().with_se_kind(AnalyticSeKind::Cluster).with_cluster_ids(clusters.clone()));
    assert!(
        (clustered - cluster_oracle).abs() < 1e-9 * (1.0 + cluster_oracle),
        "cluster {clustered} vs {cluster_oracle}"
    );
    let serial = fit(base().with_se_kind(AnalyticSeKind::NeweyWest { lag }));
    assert!(
        (serial - newey_west_oracle).abs() < 1e-9 * (1.0 + newey_west_oracle),
        "newey-west {serial} vs {newey_west_oracle}"
    );
    assert!((independent - clustered).abs() > 1e-3 * independent, "clustering must matter");
}

#[test]
fn a_fully_compelled_class_is_the_one_completion_case_and_gives_the_dag_answer() {
    // Every edge of the model is compelled, so the class holds the single completion that is the
    // DAG; the class route and the DAG route then publish the same total effect, the closed form 11.
    let on_dag = plain("dag", &average_effect()).unwrap().estimate().as_effect().unwrap().ate;
    let on_cpdag = plain("cpdag", &average_effect()).unwrap().estimate().as_effect().unwrap().ate;
    assert!((on_dag - 11.0).abs() < 0.3, "dag {on_dag}");
    assert!((on_dag - on_cpdag).abs() < 1e-9, "dag {on_dag} vs fully compelled class {on_cpdag}");
}

#[test]
fn the_class_average_effect_refuses_a_graph_that_does_not_fit_the_table_and_an_invalid_query() {
    let build = |graph: Cpdag, query: CausalQuery| {
        Study::tabular(table(&columns())).graph(graph).query(query).build().and_then(|study| study.run(&ctx()))
    };
    // A class over five variables does not describe a four-column table.
    match build(cpdag(5), average_effect()) {
        Err(CausalError::SchemaMismatch { detail }) => {
            assert!(detail.contains("graph has 5 nodes but data has 4 variables"), "{detail}");
        }
        other => panic!("a graph of the wrong size must be refused, got {:?}", other.err()),
    }
    // A treatment the class does not contain.
    let absent = CausalQuery::AverageEffect(AverageEffectQuery::binary_ate(v(9), v(2)));
    match build(cpdag(4), absent) {
        Err(CausalError::Compile { message }) => {
            assert!(message.contains("query variables not in CPDAG"), "{message}");
        }
        other => panic!("an absent treatment must be refused, got {:?}", other.err()),
    }
    // A treatment that is also the outcome.
    let same = CausalQuery::AverageEffect(AverageEffectQuery::binary_ate(v(0), v(0)));
    let error = build(cpdag(4), same).err().expect("a refusal");
    assert!(error.to_string().contains("treatment and outcome are the same variable"), "{error}");
}

#[test]
fn the_class_average_effect_refuses_a_treatment_that_never_visits_a_contrasted_level() {
    // With a = 1 on every row the control level 0 is a regime the data never visited; the contrast
    // is refused on the class route and on the DAG route alike, never read off the intercept.
    for class in ["cpdag", "dag"] {
        let error = run(class, table(&design(true, false)), average_effect(), None, None)
            .err()
            .unwrap_or_else(|| panic!("{class}: a constant treatment was estimated"));
        assert_eq!(error.reason_code(), Some("arm_not_populated"), "{class}: {error}");
        assert!(error.to_string().contains("control 0"), "{class}: {error}");
    }
}

#[test]
fn the_class_average_effect_refuses_a_contrast_level_a_two_valued_treatment_never_took() {
    // A 0/1 treatment queried at (2, -3) or (7.5, 0) used to publish the dose slope times the
    // distance (26.7, 100, -40) on every class; the unvisited level has no rows to estimate from.
    let at = |i: usize| i as f64;
    let z: Vec<f64> = (0..N).map(|i| (at(i) * 0.37).sin()).collect();
    let a: Vec<f64> = (0..N).map(|i| f64::from(u8::from(z[i] + (at(i) * 0.71).cos() > 0.0))).collect();
    let y: Vec<f64> = (0..N).map(|i| 3.0 * a[i] + z[i] + 0.1 * (at(i) * 0.31).sin()).collect();
    let data = table(&[("a", a), ("y", y), ("z", z)]);
    let ctx = ctx();
    for class in ["dag", "admg", "cpdag"] {
        let mut graph = Dag::with_variables(3);
        for (from, to) in [(2_u32, 0_u32), (2, 1), (0, 1)] {
            graph.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
        }
        let mut latent_free = Admg::with_variables(3);
        for (from, to) in [(2_u32, 0_u32), (2, 1), (0, 1)] {
            latent_free
                .insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to))
                .unwrap();
        }
        let query = |active: f64, control: f64| {
            AverageEffectQuery::with_levels(v(0), v(1), control, active)
        };
        let run = |active: f64, control: f64| {
            let builder = Study::tabular(data.clone());
            let builder = match class {
                "dag" => builder.graph(graph.clone()),
                "admg" => builder.graph(latent_free.clone()),
                _ => builder.graph(Cpdag::from_dag(&graph)),
            };
            builder
                .query(query(active, control))
                .refute(RefuteSuite::None)
                .bootstrap_replicates(0)
                .build()
                .and_then(|study| study.run(&ctx))
        };
        let sound = run(1.0, 0.0).unwrap();
        assert!(sound.estimate().as_effect().unwrap().ate > 2.0, "{class}");
        for (active, control) in [(2.0, -3.0), (7.5, 0.0), (0.0, -3.0)] {
            let error = run(active, control)
                .err()
                .unwrap_or_else(|| panic!("{class} ({active}, {control}): extrapolated"));
            assert_eq!(error.reason_code(), Some("arm_not_populated"), "{class}: {error}");
            assert!(error.to_string().contains("the treatment takes the levels 0, 1"), "{error}");
        }
    }
}
