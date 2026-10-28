//! Honest causal forest against an exact-rational tree reference with a ported random stream.
//!
//! `oracle_data/estimate_causal_forest_reference.py` (frozen in the JSON beside it) ports the
//! library's SplitMix64 stream derivation, replays each tree's subsample shuffle, split search
//! (every candidate cut, the squared-gap score, the minimum-leaf and two-arm admissibility),
//! honest leaf estimation with the parent fallback, and the per-row forest average and leaf
//! dispersion, all in `Fraction` arithmetic. The test first asserts that the library's own
//! stream equals the frozen draws, then that the product's portable trees (split features,
//! thresholds, leaf values), per-row effects and leaf dispersion equal the reference, and that
//! the marginal effect is exactly the cross-fitted AIPW estimate of the DML estimator it names.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#[path = "oracle_data/table_support.rs"]
mod table_support;

use std::sync::Arc;

use antecedent_core::{
    AssumptionSet, AverageEffectQuery, ExecutionContext, RoleHint, StreamDomain, derive_stream,
};
use antecedent_estimate::{CausalForest, ClusterDml, DmlAte};
use antecedent_expr::{ExprId, IdentifiedEstimand};
use antecedent_learn::{PredictionMap, PredictionNode};
use serde_json::Value;
use table_support::{id, table};

fn cases() -> Vec<Value> {
    serde_json::from_str::<Value>(include_str!("oracle_data/estimate_causal_forest_reference.json"))
        .unwrap()["cases"]
        .as_array()
        .unwrap()
        .clone()
}

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn close(got: f64, want: f64, rel: f64, what: &str) {
    close_with_floor(got, want, rel, 0.0, what);
}

/// `floor` is the absolute rounding of the data themselves (a few units in the last place of the
/// largest outcome), below which a difference of two large arm means cannot be resolved.
fn close_with_floor(got: f64, want: f64, rel: f64, floor: f64, what: &str) {
    assert!(
        (got - want).abs() <= rel * want.abs().max(1e-9) + floor,
        "{what}: got {got} want {want}"
    );
}

struct Problem {
    data: antecedent_data::TabularData,
    estimand: IdentifiedEstimand,
    query: AverageEffectQuery,
    n: usize,
}

fn problem(case: &Value) -> Problem {
    let (t, y) = (floats(&case["t"]), floats(&case["y"]));
    let x: Vec<Vec<f64>> = case["x"].as_array().unwrap().iter().map(floats).collect();
    let p = x[0].len();
    let n = y.len();
    let mut columns =
        vec![("t", RoleHint::TreatmentCandidate, t), ("y", RoleHint::OutcomeCandidate, y)];
    for k in 0..p {
        columns.push(("x", RoleHint::Context, x.iter().map(|row| row[k]).collect()));
    }
    let adjustment: Vec<_> = (0..p).map(|k| id(2 + u32::try_from(k).unwrap())).collect();
    Problem {
        data: table(&columns),
        estimand: IdentifiedEstimand::backdoor(
            "backdoor.adjustment",
            Arc::from(adjustment),
            ExprId::from_raw(0),
        ),
        query: AverageEffectQuery::binary_ate(id(0), id(1)),
        n,
    }
}

fn forest_of(case: &Value) -> CausalForest {
    let mut forest = CausalForest::new()
        .with_n_trees(case["n_trees"].as_u64().unwrap() as usize)
        .with_min_leaf(case["min_leaf"].as_u64().unwrap() as usize)
        .with_max_depth(case["max_depth"].as_u64().unwrap() as usize)
        .with_honesty(case["honesty"].as_bool().unwrap())
        .with_max_cuts(case["max_cuts"].as_u64().map(|c| c as usize));
    if !case["clusters"].is_null() {
        let labels: Vec<u32> = case["clusters"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        forest = forest.with_cluster_dml(ClusterDml::new(10).unwrap()).with_cluster_ids(labels);
    }
    forest
}

fn portable(effect: &antecedent_estimate::EffectEstimate) -> Vec<Vec<PredictionNode>> {
    let model = effect.heterogeneity.fitted_effect.as_ref().expect("portable forest");
    match &model.predictor.model {
        PredictionMap::Trees { trees, .. } => trees.clone(),
        other => panic!("not a tree ensemble: {other:?}"),
    }
}

#[test]
fn the_library_stream_equals_the_frozen_reference_draws() {
    for case in cases() {
        for run in case["runs"].as_array().unwrap() {
            let seed = run["seed"].as_u64().unwrap();
            let ctx = ExecutionContext::for_tests(seed);
            for (b, frozen) in run["draws"].as_array().unwrap().iter().enumerate() {
                let mut rng = ctx
                    .rng
                    .stream_for(StreamDomain::Estimate, derive_stream(0xC0F0_0000, b as u64));
                for (k, want) in frozen.as_array().unwrap().iter().enumerate() {
                    let want: u64 = want.as_str().unwrap().parse().unwrap();
                    assert_eq!(
                        rng.next_u64(),
                        want,
                        "{} seed {seed} tree {b} draw {k}",
                        case["name"]
                    );
                }
            }
        }
    }
}

#[test]
fn trees_effects_and_dispersion_match_the_exact_reference() {
    for case in cases() {
        let name = case["name"].as_str().unwrap();
        let pb = problem(&case);
        let forest = forest_of(&case);
        let floor =
            8.0 * f64::EPSILON * floats(&case["y"]).iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        for run in case["runs"].as_array().unwrap() {
            let seed = run["seed"].as_u64().unwrap();
            let prepared = forest.prepare(&pb.data, &pb.estimand, &pb.query).unwrap();
            let ctx = ExecutionContext::for_tests(seed);
            let fit = forest.fit(&prepared, &ctx, AssumptionSet::new());
            if run["refused_no_estimate"].as_bool().unwrap() {
                let message =
                    fit.err().unwrap_or_else(|| panic!("{name}: expected refusal")).to_string();
                assert!(message.contains("no honest two-arm estimate"), "{name}: {message}");
                continue;
            }
            let effect = fit.unwrap_or_else(|e| panic!("{name} seed {seed}: {e:?}"));

            // Tree structure: preorder nodes, split features and thresholds, honest leaf values.
            let trees = portable(&effect);
            let want_trees = run["trees"].as_array().unwrap();
            assert_eq!(trees.len(), want_trees.len(), "{name}: tree count");
            for (b, (got, want)) in trees.iter().zip(want_trees).enumerate() {
                let want = want.as_array().unwrap();
                assert_eq!(got.len(), want.len(), "{name} tree {b}: node count");
                for (k, (g, w)) in got.iter().zip(want).enumerate() {
                    match g {
                        PredictionNode::Split { feature, threshold, left, right, .. } => {
                            assert_eq!(
                                *feature as u64,
                                w["feature"].as_u64().unwrap(),
                                "{name} {b}/{k}"
                            );
                            assert_eq!(
                                *threshold,
                                w["threshold"].as_f64().unwrap(),
                                "{name} {b}/{k}"
                            );
                            assert_eq!(*left as u64, w["left"].as_u64().unwrap(), "{name} {b}/{k}");
                            assert_eq!(
                                *right as u64,
                                w["right"].as_u64().unwrap(),
                                "{name} {b}/{k}"
                            );
                        }
                        PredictionNode::Leaf { value } => match (value, w["leaf"].as_f64()) {
                            (Some(g), Some(w)) => close_with_floor(
                                *g,
                                w,
                                1e-9,
                                floor,
                                &format!("{name} {b}/{k} leaf"),
                            ),
                            (None, None) => {}
                            (g, w) => panic!("{name} {b}/{k}: leaf {g:?} vs {w:?}"),
                        },
                    }
                }
            }

            // Per-row forest effect and leaf dispersion.
            let cate = effect.heterogeneity.cate.as_ref().expect("cate");
            let want_cate = floats(&run["cate"]);
            assert_eq!(cate.len(), pb.n);
            for (i, (g, w)) in cate.iter().zip(&want_cate).enumerate() {
                close_with_floor(*g, *w, 1e-9, floor, &format!("{name} seed {seed} cate[{i}]"));
            }
            match (&effect.heterogeneity.cate_leaf_dispersion, run["dispersion"].as_array()) {
                (Some(got), Some(want)) => {
                    for (i, (g, w)) in got.iter().zip(want).enumerate() {
                        close(*g, w.as_f64().unwrap(), 1e-6, &format!("{name} dispersion[{i}]"));
                    }
                }
                (None, None) => {}
                (got, want) => panic!("{name}: dispersion {got:?} vs {want:?}"),
            }
            assert!(
                effect.heterogeneity.cate_se.is_none(),
                "{name}: a forest publishes no CATE SE"
            );

            // The marginal effect is the cross-fitted AIPW estimate of the DML estimator.
            let mut dml = DmlAte::new().with_folds(5.min(pb.n));
            dml.units = forest.units.clone();
            let dml_problem = dml.prepare(&pb.data, &pb.estimand, &pb.query).unwrap();
            let marginal = dml.fit(&dml_problem, &ctx, AssumptionSet::new()).unwrap();
            assert_eq!(effect.ate, marginal.ate, "{name}: marginal effect");
            assert_eq!(effect.se_analytic(), marginal.se_analytic(), "{name}: marginal se");
        }
    }
}

#[test]
fn a_table_of_two_minimum_leaves_without_honesty_does_not_depend_on_the_seed() {
    let all = cases();
    let case = all.iter().find(|c| c["name"] == "half_table_without_honesty_is_seed_free").unwrap();
    let pb = problem(case);
    let forest = forest_of(case);
    let mut reference: Option<(Vec<f64>, Vec<Vec<PredictionNode>>)> = None;
    for seed in [1u64, 7, 99, 12_345] {
        let prepared = forest.prepare(&pb.data, &pb.estimand, &pb.query).unwrap();
        let effect = forest
            .fit(&prepared, &ExecutionContext::for_tests(seed), AssumptionSet::new())
            .unwrap();
        let cate = effect.heterogeneity.cate.as_ref().unwrap().to_vec();
        let trees = portable(&effect);
        match &reference {
            None => reference = Some((cate, trees)),
            Some((c0, t0)) => {
                for (a, b) in cate.iter().zip(c0) {
                    close(*a, *b, 1e-12, "cate across seeds");
                }
                for (ta, tb) in trees.iter().zip(t0) {
                    for (a, b) in ta.iter().zip(tb) {
                        match (a, b) {
                            (
                                PredictionNode::Split { feature: fa, threshold: xa, .. },
                                PredictionNode::Split { feature: fb, threshold: xb, .. },
                            ) => assert_eq!((fa, xa), (fb, xb)),
                            (
                                PredictionNode::Leaf { value: Some(a) },
                                PredictionNode::Leaf { value: Some(b) },
                            ) => {
                                close(*a, *b, 1e-12, "leaf across seeds");
                            }
                            _ => panic!("tree shape depends on the seed"),
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn settings_and_declarations_the_forest_cannot_honour_are_refused() {
    let all = cases();
    let case = &all[0];
    let pb = problem(case);
    let fit = |forest: CausalForest| {
        let prepared = forest.prepare(&pb.data, &pb.estimand, &pb.query)?;
        forest.fit(&prepared, &ExecutionContext::for_tests(1), AssumptionSet::new())
    };
    let base = || forest_of(case);
    assert!(fit(base().with_n_trees(0)).is_err(), "zero trees");
    assert!(fit(base().with_min_leaf(0)).is_err(), "zero minimum leaf");
    assert!(fit(base().with_max_depth(0)).is_err(), "zero depth");
    assert!(fit(base().with_max_cuts(Some(0))).is_err(), "zero cuts");
    assert!(fit(base().with_min_leaf(31)).is_err(), "fewer than two minimum leaves of rows");
    // Cluster labels need the declaration and the declaration needs labels.
    assert!(fit(base().with_cluster_ids(vec![0; pb.n])).is_err(), "labels without a unit");
    assert!(
        fit(base().with_cluster_dml(ClusterDml::new(10).unwrap())).is_err(),
        "a unit without labels"
    );
    let labels: Vec<u32> = (0..pb.n as u32).map(|i| i % 12).collect();
    assert!(
        fit(base()
            .with_cluster_dml(ClusterDml::new(10).unwrap())
            .with_cluster_ids(labels.clone())
            .with_source_cluster_ids(labels))
        .is_err(),
        "labels declared both per analysed row and per source row"
    );
    // A dyadic unit has no whole-cluster subsample.
    let mut dyadic = base().with_cluster_dml(ClusterDml::new(10).unwrap());
    dyadic.units.first = Some((0..pb.n as u32).map(|i| i % 12).collect());
    dyadic.units.second = Some((0..pb.n as u32).map(|i| i % 11).collect());
    assert!(fit(dyadic).is_err(), "dyadic unit");
}
