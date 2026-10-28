//! Do-samplers (weighting, kernel density, Metropolis) against definition-level references.
//!
//! `oracle_data/gcm_sampling_reference.py` (frozen in the JSON beside it, computed at 50 digits
//! with mpmath) evaluates, from the definitions, the Silverman normal-reference bandwidth, the
//! Gaussian kernel density, the root-treatment matching window and the confounded kernel/density
//! importance weights with the 1e6 cap, normalisation, Kish effective sample size and weighted
//! mean, in five regimes: a central do-value, a far tail (degenerate weights), a narrow fitted
//! conditional that triggers the cap, a treatment narrower than the kernel-width floor, and a
//! 1e6 offset. This test fits no model: it sets the treatment's linear-Gaussian mechanism to
//! the reference parameters and requires the sampler to reproduce the reference. The Metropolis
//! chain has no closed form, so it is checked on the one property its construction implies: its
//! stationary law is the kernel density of the pilot draws, whose mean and variance are the
//! pilot's mean and the pilot's variance plus the squared bandwidth.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    CausalRng, CausalSchemaBuilder, ExecutionContext, Intervention, MeasurementSpec, RoleHint,
    SmallRoleSet, Value, ValueType, VariableId,
};
use antecedent_data::column::{Float64Column, ValidityBitmap};
use antecedent_data::{OwnedColumn, OwnedColumnarStorage, TabularData};
use antecedent_graph::{Dag, DenseNodeId};
use antecedent_model::{
    CompiledCausalModel, CompiledMechanismStore, DoSampleResult, KdeDoSampler, McmcDoSampler,
    MechanismSlot, MechanismWorkspace, WeightingDoSampler,
};
use serde_json::Value as Json;

fn reference() -> Json {
    serde_json::from_str(include_str!("oracle_data/gcm_sampling_reference.json")).unwrap()
}

fn floats(v: &Json) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn near(got: f64, want: f64, rel: f64, what: &str) {
    assert!((got - want).abs() <= rel * want.abs().max(1e-300), "{what}: {got} vs {want}");
}

fn linear(intercept: f64, coeffs: &[f64], sigma: f64) -> MechanismSlot {
    MechanismSlot::LinearGaussian { intercept, coeffs: Arc::from(coeffs.to_vec()), sigma }
}

fn table(columns: &[(&str, &[f64])]) -> TabularData {
    let mut b = CausalSchemaBuilder::new();
    for (name, _) in columns {
        b.add_variable(
            *name,
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
    }
    let schema = b.build().unwrap();
    let n = columns[0].1.len();
    let validity = ValidityBitmap::all_valid(n);
    let cols = columns
        .iter()
        .enumerate()
        .map(|(i, (_, v))| {
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(u32::try_from(i).unwrap()),
                    Arc::from(v.to_vec()),
                    validity.clone(),
                )
                .unwrap(),
            )
        })
        .collect();
    TabularData::new(OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap())
}

fn model(edges: &[(u32, u32)], n: u32, slots: Vec<MechanismSlot>) -> CompiledCausalModel {
    let mut g = Dag::with_variables(n);
    for &(a, b) in edges {
        g.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
    }
    CompiledCausalModel::compile(g)
        .unwrap()
        .with_mechanisms(CompiledMechanismStore { slots: Arc::from(slots) })
}

#[test]
fn confounded_weights_ess_and_mean_match_the_definition() {
    let ctx = ExecutionContext::for_tests(1);
    for case in reference()["confounded"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let (z, t, y) = (floats(&case["z"]), floats(&case["t"]), floats(&case["y"]));
        let m = model(
            &[(0, 1), (0, 2), (1, 2)],
            3,
            vec![
                linear(0.0, &[], 1.0),
                linear(
                    case["intercept"].as_f64().unwrap(),
                    &[case["coef"].as_f64().unwrap()],
                    case["sigma"].as_f64().unwrap(),
                ),
                linear(0.0, &[0.0, 0.0], 1.0),
            ],
        );
        let data = table(&[("z", &z), ("t", &t), ("y", &y)]);
        let sampler = WeightingDoSampler::new(VariableId::from_raw(1), VariableId::from_raw(2));
        let got = sampler.estimate(&m, &data, case["t0"].as_f64().unwrap(), &ctx).unwrap();
        near(got.bandwidth.unwrap(), case["bandwidth"].as_f64().unwrap(), 1e-12, name);
        for (g, w) in got.weights.iter().zip(floats(&case["weights"])) {
            near(*g, w, 1e-9, name);
        }
        near(got.ess.unwrap(), case["ess"].as_f64().unwrap(), 1e-9, name);
        near(WeightingDoSampler::weighted_mean(&got), case["mean"].as_f64().unwrap(), 1e-9, name);
        assert_eq!(
            got.notes.iter().any(|n| n.contains("degenerate weighting")),
            case["degenerate"].as_bool().unwrap(),
            "{name}"
        );
        assert_eq!(got.values.to_vec(), y, "{name}: every unit is retained");
    }
}

#[test]
fn the_weighting_sampler_honours_cancellation_and_the_memory_limit() {
    use antecedent_model::ModelError;
    let (z, t, y) = ([0.0, 1.0, 2.0, 3.0], [0.1, 0.9, 2.2, 2.8], [1.0, 2.0, 3.0, 4.0]);
    let m = model(
        &[(0, 1), (0, 2), (1, 2)],
        3,
        vec![linear(0.0, &[], 1.0), linear(0.0, &[1.0], 1.0), linear(0.0, &[0.0, 0.0], 1.0)],
    );
    let data = table(&[("z", &z), ("t", &t), ("y", &y)]);
    let sampler = WeightingDoSampler::new(VariableId::from_raw(1), VariableId::from_raw(2));
    assert!(sampler.estimate(&m, &data, 1.0, &ExecutionContext::for_tests(1)).is_ok());

    let cancelled = ExecutionContext::for_tests(1);
    cancelled.cancellation.cancel();
    let err = sampler.estimate(&m, &data, 1.0, &cancelled).unwrap_err();
    assert!(matches!(err, ModelError::Cancelled), "{err:?}");

    let tight = ExecutionContext::for_tests(1).with_memory_limit(Some(8));
    let err = sampler.estimate(&m, &data, 1.0, &tight).unwrap_err();
    assert!(matches!(err, ModelError::ResourceLimit { .. }), "{err:?}");

    // The root branch is bound by the same account.
    let root = model(&[(0, 1)], 2, vec![linear(0.0, &[], 1.0), linear(0.0, &[0.0], 1.0)]);
    let rdata = table(&[("t", &t), ("y", &y)]);
    let rs = WeightingDoSampler::new(VariableId::from_raw(0), VariableId::from_raw(1));
    let err = rs.estimate(&root, &rdata, 0.1, &cancelled).unwrap_err();
    assert!(matches!(err, ModelError::Cancelled), "{err:?}");
    let err = rs.estimate(&root, &rdata, 0.1, &tight).unwrap_err();
    assert!(matches!(err, ModelError::ResourceLimit { .. }), "{err:?}");
}

#[test]
fn the_root_matching_window_selects_exactly_the_units_the_definition_selects() {
    let ctx = ExecutionContext::for_tests(1);
    for case in reference()["root"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let (t, y) = (floats(&case["t"]), floats(&case["y"]));
        let m = model(&[(0, 1)], 2, vec![linear(0.0, &[], 1.0), linear(0.0, &[0.0], 1.0)]);
        let data = table(&[("t", &t), ("y", &y)]);
        let sampler = WeightingDoSampler::new(VariableId::from_raw(0), VariableId::from_raw(1))
            .with_root_match_tolerance(case["tolerance"].as_f64().unwrap());
        let want = floats(&case["selected"]);
        let got = sampler.estimate(&m, &data, case["t0"].as_f64().unwrap(), &ctx);
        if want.is_empty() {
            assert!(got.is_err(), "{name}: no unit in the window is a refusal");
            continue;
        }
        let got = got.unwrap();
        assert_eq!(got.values.to_vec(), want, "{name}");
        assert!(got.weights.iter().all(|w| *w == 1.0), "{name}: unit weights");
    }
    // A negative or non-finite window is refused, not treated as exact matching.
    let data = table(&[("t", &[0.0, 1.0]), ("y", &[1.0, 2.0])]);
    let m = model(&[(0, 1)], 2, vec![linear(0.0, &[], 1.0), linear(0.0, &[0.0], 1.0)]);
    for bad in [-1.0, f64::NAN, f64::INFINITY] {
        let sampler = WeightingDoSampler::new(VariableId::from_raw(0), VariableId::from_raw(1))
            .with_root_match_tolerance(bad);
        assert!(sampler.estimate(&m, &data, 1.0, &ExecutionContext::for_tests(1)).is_err());
    }
}

#[test]
fn the_kernel_density_matches_the_definition_including_the_bandwidth_floor() {
    for case in reference()["density"].as_array().unwrap() {
        let result = DoSampleResult {
            values: Arc::from(floats(&case["values"])),
            weights: Arc::from([]),
            method: Arc::from("kde"),
            notes: Vec::new(),
            accept_rate: None,
            bandwidth: case["bandwidth"].as_f64(),
            ess: None,
        };
        for (x, want) in floats(&case["points"]).into_iter().zip(floats(&case["density"])) {
            let got = KdeDoSampler::density(&result, x);
            assert!(
                (got - want).abs() <= 1e-12 * want.abs().max(1e-300) + 1e-300,
                "bandwidth {:?} at {x}: {got} vs {want}",
                case["bandwidth"]
            );
        }
    }
}

#[test]
fn the_kde_sampler_reports_the_silverman_bandwidth_of_its_own_draws() {
    let m = model(&[(0, 1)], 2, vec![linear(0.0, &[], 1.0), linear(1.0, &[2.0], 0.5)]);
    let ctx = ExecutionContext::for_tests(1);
    let iv = [Intervention::set(VariableId::from_raw(0), Value::f64(3.0))];
    let mut rng = CausalRng::from_seed(5);
    let got = KdeDoSampler::new(VariableId::from_raw(1))
        .sample(&m, &iv, 400, &mut rng, &mut MechanismWorkspace::default(), &ctx)
        .unwrap();
    let n = got.values.len() as f64;
    let mean = got.values.iter().sum::<f64>() / n;
    let var = got.values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let want = var.sqrt() * (4.0 / (3.0 * n)).powf(0.2);
    near(got.bandwidth.unwrap(), want, 1e-12, "normal-reference rule");
    // The draws are the interventional law: y = 1 + 2 * 3 + noise with sd 0.5.
    assert!((mean - 7.0).abs() < 0.2, "{mean}");
    // A caller bandwidth is used as given.
    let mut sampler = KdeDoSampler::new(VariableId::from_raw(1));
    sampler.bandwidth = Some(0.125);
    let fixed = sampler
        .sample(&m, &iv, 50, &mut CausalRng::from_seed(5), &mut MechanismWorkspace::default(), &ctx)
        .unwrap();
    assert_eq!(fixed.bandwidth, Some(0.125));
}

#[test]
fn the_silverman_bandwidth_reference_agrees_on_the_frozen_samples() {
    // The weighting sampler reports the shared rule floored at 1e-8; the frozen samples cover a
    // 1e6 offset (where the variance is computed from centred sums) and a wide range.
    let ctx = ExecutionContext::for_tests(1);
    let m = model(
        &[(0, 1), (0, 2), (1, 2)],
        3,
        vec![linear(0.0, &[], 1.0), linear(0.0, &[0.0], 1.0), linear(0.0, &[0.0, 0.0], 1.0)],
    );
    for case in reference()["bandwidth_samples"].as_array().unwrap() {
        let t = floats(&case["x"]);
        let z = vec![0.0; t.len()];
        let data = table(&[("z", &z), ("t", &t), ("y", &z)]);
        let sampler = WeightingDoSampler::new(VariableId::from_raw(1), VariableId::from_raw(2));
        let got = sampler.estimate(&m, &data, t[0], &ctx).unwrap();
        near(got.bandwidth.unwrap(), case["h"].as_f64().unwrap().max(1e-8), 1e-9, "bandwidth");
    }
}

#[test]
fn the_metropolis_chain_has_the_kernel_density_of_its_pilot_as_stationary_law() {
    let m = model(&[(0, 1)], 2, vec![linear(0.0, &[], 1.0), linear(1.0, &[2.0], 1.0)]);
    let ctx = ExecutionContext::for_tests(1);
    let iv = [Intervention::set(VariableId::from_raw(0), Value::f64(3.0))];
    let n = 6000;
    // The chain draws its pilot first from the same stream, so the pilot is reproducible.
    let mut pilot_rng = CausalRng::from_seed(21);
    let pilot = antecedent_model::sample_interventional(
        &m,
        &iv,
        n,
        &mut pilot_rng,
        &mut MechanismWorkspace::default(),
        &ctx,
    )
    .unwrap();
    let col = pilot.column(1).unwrap();
    let count = col.len() as f64;
    let pilot_mean = col.iter().sum::<f64>() / count;
    let pilot_var = col.iter().map(|v| (v - pilot_mean).powi(2)).sum::<f64>() / (count - 1.0);
    let h2 = pilot_var * (4.0 / (3.0 * count)).powf(0.4);
    let kde_var = pilot_var * (count - 1.0) / count + h2;

    let mut sampler = McmcDoSampler::new(VariableId::from_raw(1));
    sampler.proposal_sd = 1.0;
    let out = sampler
        .sample(&m, &iv, n, &mut CausalRng::from_seed(21), &mut MechanismWorkspace::default(), &ctx)
        .unwrap();
    assert_eq!(out.values.len(), n);
    let mean = out.values.iter().sum::<f64>() / n as f64;
    let var = out.values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0);
    let rate = out.accept_rate.unwrap();
    assert!(rate > 0.2 && rate < 0.95, "{rate}");
    assert!((mean - pilot_mean).abs() < 0.15, "chain mean {mean} vs {pilot_mean}");
    assert!((var / kde_var - 1.0).abs() < 0.2, "chain variance {var} vs {kde_var}");
}
