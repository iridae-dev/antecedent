//! Compiled structural model and noise replay against exact evaluation of the structural
//! equations.
//!
//! `oracle_data/scm_replay_reference.py` (frozen in the JSON beside it) fixes four linear
//! structural models whose node ids run against the topological order (a diamond with a
//! collider, the same shape with intercepts near 1e5 and coefficients from 2^-20 to 2^6, a deep
//! reversed chain, and a five-parent collider with an isolated node) together with exact noise
//! tables, and evaluates every node by the structural equation in rational arithmetic under
//! seven overlays: none, a hard set, shifts, a hard set beside shifts of other nodes, two hard
//! sets, a soft linear override, and a mixed hard set, shift and soft override. This test supplies
//! the same noise to `NoiseReplayer::replay` (one table, and a coalition of two tables under a
//! node mask) and requires the compiled evaluation to match, with a tolerance scaled to the
//! magnitude of the terms. It also drives the ancestral sampler with a vanishing noise scale and
//! requires the deterministic part to match, and checks that a node both hard-set and shifted is
//! refused.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{CausalRng, ExecutionContext, MechanismOverride};
use antecedent_graph::{Dag, DenseNodeId};
use antecedent_model::{
    CompiledCausalModel, CompiledMechanismStore, InterventionOverlay, MechanismSlot,
    MechanismWorkspace, ModelView, NoiseReplayer, NoiseSource, NoiseTable, sample_with_overlay,
};
use serde_json::Value;

/// Noise scale that puts the stochastic part far below the tolerance of every magnitude here.
const TINY_SIGMA: f64 = 8.881784197001252e-16;

fn node(i: usize) -> DenseNodeId {
    DenseNodeId::from_raw(u32::try_from(i).unwrap())
}

fn num(v: &Value) -> f64 {
    v.as_f64().unwrap()
}

/// Coefficients of node `i` in the order the compiled model gathers its parents.
fn gathered(model: &CompiledCausalModel, i: usize, by_parent: &Value) -> Vec<f64> {
    model
        .gather_for(node(i))
        .unwrap()
        .parents
        .iter()
        .map(|p| num(&by_parent[p.as_usize().to_string()]))
        .collect()
}

fn compile(sc: &Value, sigma: f64) -> CompiledCausalModel {
    let n = sc["n"].as_u64().unwrap() as usize;
    let mut graph = Dag::with_variables(u32::try_from(n).unwrap());
    for e in sc["edges"].as_array().unwrap() {
        graph
            .insert_directed(
                node(e[0].as_u64().unwrap() as usize),
                node(e[1].as_u64().unwrap() as usize),
            )
            .unwrap();
    }
    let model = CompiledCausalModel::compile(graph).unwrap();
    let slots: Vec<MechanismSlot> = (0..n)
        .map(|i| MechanismSlot::LinearGaussian {
            intercept: num(&sc["intercept"][i]),
            coeffs: Arc::from(gathered(&model, i, &sc["coef"][i])),
            sigma,
        })
        .collect();
    model.with_mechanisms(CompiledMechanismStore { slots: Arc::from(slots) })
}

fn overlay(
    model: &CompiledCausalModel,
    case: &Value,
    n: usize,
    soft_sigma: f64,
) -> InterventionOverlay {
    let mut o = InterventionOverlay::observational(n);
    for (k, v) in case["hard"].as_object().unwrap() {
        o.hard_set[k.parse::<usize>().unwrap()] = Some(num(v));
    }
    for (k, v) in case["shift"].as_object().unwrap() {
        o.shifts[k.parse::<usize>().unwrap()] = num(v);
    }
    for (k, v) in case["soft"].as_object().unwrap() {
        let i = k.parse::<usize>().unwrap();
        let mut params = vec![num(&v["intercept"])];
        params.extend(gathered(model, i, &v["coef"]));
        params.push(soft_sigma);
        o.soft[i] = Some(MechanismOverride::named("linear_gaussian", params));
    }
    o
}

fn table(model: &CompiledCausalModel, rows: &Value, n: usize) -> NoiseTable {
    let r = rows.as_array().unwrap().len();
    let mut values = vec![0.0; n * r];
    for (row, line) in rows.as_array().unwrap().iter().enumerate() {
        for i in 0..n {
            values[i * r + row] = num(&line[i]);
        }
    }
    NoiseTable::from_abducted(model, r, values, &vec![false; n]).unwrap()
}

fn close(got: f64, want: f64, magnitude: f64, what: &str) {
    assert!((got - want).abs() <= 1e-13 * magnitude.max(1.0), "{what}: {got} vs {want}");
}

#[test]
fn compiled_evaluation_and_replay_match_the_structural_equations_exactly() {
    let scenarios: Value =
        serde_json::from_str(include_str!("oracle_data/scm_replay_reference.json")).unwrap();
    let ctx = ExecutionContext::for_tests(5);
    for sc in scenarios.as_array().unwrap() {
        let name = sc["name"].as_str().unwrap();
        let n = sc["n"].as_u64().unwrap() as usize;
        let model = compile(sc, 1.0);
        let t0 = table(&model, &sc["noise"][0], n);
        let t1 = table(&model, &sc["noise"][1], n);
        let mut replayer = NoiseReplayer::new();
        for (c, case) in sc["cases"].as_array().unwrap().iter().enumerate() {
            let what = format!("{name} case {c}");
            let view = ModelView::with_overlay(&model, overlay(&model, case, n, 1.0));
            let got = replayer.replay(&view, NoiseSource::Table(&t0), 0, &ctx).unwrap();
            let mask: Vec<bool> =
                case["mask"].as_array().unwrap().iter().map(|b| b.as_bool().unwrap()).collect();
            let coalition = NoiseSource::coalition(&t0, &t1, &mask).unwrap();
            let mixed = replayer.replay(&view, coalition, 0, &ctx).unwrap();
            for i in 0..n {
                for (r, (want, mag)) in case["values"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .zip(case["magnitude"].as_array().unwrap())
                    .enumerate()
                {
                    close(got.column(i).unwrap()[r], num(&want[i]), num(&mag[i]), &what);
                }
                for (r, (want, mag)) in case["coalition_values"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .zip(case["coalition_magnitude"].as_array().unwrap())
                    .enumerate()
                {
                    close(mixed.column(i).unwrap()[r], num(&want[i]), num(&mag[i]), &what);
                }
            }
        }
    }
}

#[test]
fn ancestral_sampling_with_a_vanishing_noise_scale_follows_the_deterministic_part() {
    let scenarios: Value =
        serde_json::from_str(include_str!("oracle_data/scm_replay_reference.json")).unwrap();
    for sc in scenarios.as_array().unwrap() {
        let name = sc["name"].as_str().unwrap();
        let n = sc["n"].as_u64().unwrap() as usize;
        let model = compile(sc, TINY_SIGMA);
        for (c, case) in sc["cases"].as_array().unwrap().iter().enumerate() {
            let what = format!("{name} case {c}");
            let view = ModelView::with_overlay(&model, overlay(&model, case, n, TINY_SIGMA));
            let mut rng = CausalRng::from_seed(9);
            let mut ws = MechanismWorkspace::default();
            let batch = sample_with_overlay(&view, 3, &mut rng, &mut ws).unwrap();
            for i in 0..n {
                for r in 0..3 {
                    close(
                        batch.column(i).unwrap()[r],
                        num(&case["deterministic"][i]),
                        num(&case["deterministic_magnitude"][i]) * 1e3,
                        &what,
                    );
                }
            }
        }
    }
}

#[test]
fn a_node_that_is_both_hard_set_and_shifted_is_refused() {
    let scenarios: Value =
        serde_json::from_str(include_str!("oracle_data/scm_replay_reference.json")).unwrap();
    let sc = &scenarios[0];
    let n = sc["n"].as_u64().unwrap() as usize;
    let model = compile(sc, 1.0);
    let mut o = InterventionOverlay::observational(n);
    o.hard_set[2] = Some(1.0);
    o.shifts[2] = 1.0;
    let view = ModelView::with_overlay(&model, o);
    assert!(NoiseTable::sample(&view, 4, &ExecutionContext::for_tests(1)).is_err());
    let mut rng = CausalRng::from_seed(1);
    assert!(sample_with_overlay(&view, 4, &mut rng, &mut MechanismWorkspace::default()).is_err());
}

#[test]
fn a_declared_activation_mask_is_refused_rather_than_ignored() {
    // `InterventionOverlay::active` is documented as a per-node activation mask. No evaluation
    // path reads it, so a node declared inactive was evaluated exactly as if active; the
    // overlay now refuses the declaration instead.
    let scenarios: Value =
        serde_json::from_str(include_str!("oracle_data/scm_replay_reference.json")).unwrap();
    let sc = &scenarios[0];
    let n = sc["n"].as_u64().unwrap() as usize;
    let model = compile(sc, 1.0);
    let mut o = InterventionOverlay::observational(n);
    o.hard_set[2] = Some(1.0);
    o.active[2] = false;
    assert!(o.validate().is_err(), "validate must refuse an activation mask");
    let view = ModelView::with_overlay(&model, o);
    assert!(NoiseTable::sample(&view, 4, &ExecutionContext::for_tests(1)).is_err());
    let mut rng = CausalRng::from_seed(1);
    assert!(sample_with_overlay(&view, 4, &mut rng, &mut MechanismWorkspace::default()).is_err());
}
