//! Frozen external and analytical oracle checks for advanced SCM mechanisms.

use std::sync::Arc;

use antecedent_core::{
    CausalRng, CausalSchemaBuilder, ExecutionContext, Intervention, MeasurementSpec, RoleHint,
    SmallRoleSet, Value, ValueType, VariableId,
};
use antecedent_data::{
    Float64Column, OwnedColumn, OwnedColumnarStorage, TabularData, ValidityBitmap,
};
use antecedent_graph::{Dag, DenseNodeId};
use antecedent_model::{
    CompiledCausalModel, CompiledMechanismStore, KdeDoSampler, McmcDoSampler, MechanismFamily,
    MechanismRegistry, MechanismSlot, MechanismWorkspace, SelectionPolicy, interventional_mean,
    kalman_filter, rts_smooth,
};
use serde_json::Value as JsonValue;

fn idx_f64(i: usize) -> f64 {
    f64::from(u32::try_from(i).expect("test index fits u32"))
}

fn json_usize(value: &JsonValue, label: &str) -> usize {
    usize::try_from(value.as_u64().expect(label)).expect("fits usize")
}

fn fixture(path: &str) -> JsonValue {
    let raw = match path {
        "mechanisms" => {
            include_str!("../../../conformance/gcm/discrete_hierarchical_glm/expected.json")
        }
        "bvar" => include_str!("../../../conformance/gcm/minnesota_bvar/expected.json"),
        "lgssm" => include_str!("../../../conformance/gcm/lgssm/expected.json"),
        "gp" => include_str!("../../../conformance/gcm/gaussian_process/expected.json"),
        "do" => include_str!("../../../conformance/gcm/do_sampling_grid/expected.json"),
        _ => panic!("unknown fixture"),
    };
    serde_json::from_str(raw).expect("fixture JSON")
}

fn table(names: &[&str], columns: Vec<Vec<f64>>) -> TabularData {
    let n = columns[0].len();
    let mut builder = CausalSchemaBuilder::new();
    for name in names {
        builder
            .add_variable(
                *name,
                ValueType::Continuous,
                SmallRoleSet::from_hint(RoleHint::Context),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
    }
    let schema = builder.build().unwrap();
    let columns = columns
        .into_iter()
        .enumerate()
        .map(|(index, values)| {
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(u32::try_from(index).expect("fit")),
                    Arc::from(values),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            )
        })
        .collect();
    TabularData::new(OwnedColumnarStorage::try_new(schema, columns, None, None).unwrap())
}

fn chain_model(data: &TabularData, family: MechanismFamily) -> CompiledCausalModel {
    let mut graph = Dag::with_variables(2);
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let compiled = CompiledCausalModel::compile(graph).unwrap();
    let registry = if matches!(family, MechanismFamily::Discrete) {
        MechanismRegistry::standard()
    } else {
        MechanismRegistry::with_bayesian_families()
    };
    let (store, _) =
        registry.assign_and_fit(&compiled, data, SelectionPolicy::RequireFamily(family)).unwrap();
    compiled.with_mechanisms(store)
}

#[test]
fn discrete_and_hierarchical_glm_match_their_independent_oracles() {
    let expected = fixture("mechanisms");
    let n = json_usize(&expected["data"]["n"], "n");
    let mut x = vec![0.0; n];
    let mut y = vec![0.0; n];
    for index in 0..n {
        let parent = if index < n / 2 { 0.0 } else { 1.0 };
        x[index] = parent;
        y[index] = if index % 8 == 0 { 1.0 - parent } else { parent };
    }
    let data = table(&["x", "y"], vec![x, y]);
    let tolerance = expected["acceptance"]["coefficient_atol"].as_f64().unwrap();

    let discrete = chain_model(&data, MechanismFamily::Discrete);
    let MechanismSlot::Discrete { logit_coeffs: Some(coefficients), .. } =
        discrete.mechanisms.get(DenseNodeId::from_raw(1))
    else {
        panic!("conditional discrete slot");
    };
    let unpenalized = expected["reference"]["unpenalized_logit_coefficients"].as_array().unwrap();
    let unpen_intercept = unpenalized[0].as_f64().unwrap();
    let unpen_slope = unpenalized[1].as_f64().unwrap();
    assert!((coefficients[2] - unpen_intercept).abs() <= tolerance);
    assert!((coefficients[3] - unpen_slope).abs() <= tolerance);

    // MM-014: EB λ must shrink the slope on non-separated data via always-on ridge,
    // not only as a separation fallback.
    let hierarchical = chain_model(&data, MechanismFamily::HierarchicalGlm);
    let MechanismSlot::Discrete { logit_coeffs: Some(coefficients), .. } =
        hierarchical.mechanisms.get(DenseNodeId::from_raw(1))
    else {
        panic!("hierarchical GLM slot");
    };
    let ridge = expected["reference"]["ridge_logit_coefficients"].as_array().unwrap();
    let ridge_intercept = ridge[0].as_f64().unwrap();
    let ridge_slope = ridge[1].as_f64().unwrap();
    assert!((coefficients[2] - ridge_intercept).abs() <= tolerance);
    assert!((coefficients[3] - ridge_slope).abs() <= tolerance);
    assert!(
        (ridge_slope - unpen_slope).abs() > 0.1,
        "fixture must keep a clear EB-ridge vs unpenalized gap"
    );
    assert!(
        (coefficients[3] - unpen_slope).abs() > 0.1,
        "HierarchicalGlm must apply EB ridge on ordinary data (got slope {})",
        coefficients[3]
    );
}

#[test]
fn minnesota_bvar_matches_independent_augmented_ridge_solve() {
    let expected = fixture("bvar");
    let n = json_usize(&expected["data"]["n"], "n");
    let mut x1 = Vec::with_capacity(n);
    let mut x2 = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for index in 0..n {
        let i = idx_f64(index);
        let first = (0.13 * i).sin();
        let second = (0.07 * i).cos();
        x1.push(first);
        x2.push(second);
        y.push(1.1 + 0.75 * first - 0.35 * second + 0.12 * (0.41 * i).sin());
    }
    let data = table(&["x1", "x2", "y"], vec![x1, x2, y]);
    let mut graph = Dag::with_variables(3);
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(2)).unwrap();
    graph.insert_directed(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)).unwrap();
    let compiled = CompiledCausalModel::compile(graph).unwrap();
    let (store, _) = MechanismRegistry::with_bayesian_families()
        .assign_and_fit(&compiled, &data, SelectionPolicy::RequireFamily(MechanismFamily::Bvar))
        .unwrap();
    let MechanismSlot::Bvar { intercept, coeffs, sigma } = store.get(DenseNodeId::from_raw(2))
    else {
        panic!("BVAR slot");
    };
    let tolerance = expected["acceptance"]["atol"].as_f64().unwrap();
    assert!((*intercept - expected["reference"]["intercept"].as_f64().unwrap()).abs() <= tolerance);
    for (actual, target) in
        coeffs.iter().zip(expected["reference"]["coefficients"].as_array().unwrap())
    {
        assert!((*actual - target.as_f64().unwrap()).abs() <= tolerance);
    }
    assert!(
        (*sigma - expected["reference"]["augmented_residual_sigma"].as_f64().unwrap()).abs()
            <= tolerance
    );
}

#[test]
fn kalman_filter_and_rts_smoother_match_clean_room_recursion() {
    let expected = fixture("lgssm");
    let p = &expected["parameters"];
    let observations: Vec<f64> = serde_json::from_value(expected["observations"].clone()).unwrap();
    let (filtered_mean, filtered_variance, predicted_mean, predicted_variance) = kalman_filter(
        &observations,
        p["a"].as_f64().unwrap(),
        p["process_variance"].as_f64().unwrap(),
        p["observation_variance"].as_f64().unwrap(),
        p["initial_mean"].as_f64().unwrap(),
        p["initial_variance"].as_f64().unwrap(),
    );
    let (smoothed_mean, smoothed_variance, lag_one_covariance) = rts_smooth(
        p["a"].as_f64().unwrap(),
        &filtered_mean,
        &filtered_variance,
        &predicted_mean,
        &predicted_variance,
    );
    let tolerance = expected["acceptance"]["atol"].as_f64().unwrap();
    for (name, actual) in [
        ("filtered_mean", filtered_mean),
        ("filtered_variance", filtered_variance),
        ("predicted_mean", predicted_mean),
        ("predicted_variance", predicted_variance),
        ("smoothed_mean", smoothed_mean),
        ("smoothed_variance", smoothed_variance),
        ("lag_one_covariance", lag_one_covariance),
    ] {
        let target = expected["reference"][name].as_array().unwrap();
        for (actual, target) in actual.iter().zip(target) {
            assert!((*actual - target.as_f64().unwrap()).abs() <= tolerance, "{name}");
        }
    }
}

fn analytic_do_model() -> CompiledCausalModel {
    let mut graph = Dag::with_variables(2);
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let compiled = CompiledCausalModel::compile(graph).unwrap();
    compiled.with_mechanisms(CompiledMechanismStore {
        slots: Arc::from([
            MechanismSlot::LinearGaussian { intercept: 0.0, coeffs: Arc::from([]), sigma: 1.0 },
            MechanismSlot::LinearGaussian { intercept: 1.0, coeffs: Arc::from([2.0]), sigma: 0.5 },
        ]),
    })
}

fn moments(values: &[f64]) -> (f64, f64) {
    let n = idx_f64(values.len());
    let mean = values.iter().sum::<f64>() / n;
    let variance = values.iter().map(|value| (value - mean).powi(2)).sum::<f64>() / n;
    (mean, variance)
}

#[test]
fn do_samplers_calibrate_to_analytic_linear_gaussian_target() {
    let expected = fixture("do");
    let model = analytic_do_model();
    let intervention = [Intervention::set(VariableId::from_raw(0), Value::f64(1.0))];
    let mean_target = expected["model"]["outcome_mean"].as_f64().unwrap();
    let variance_target = expected["model"]["outcome_variance"].as_f64().unwrap();
    let seeds = expected["acceptance"]["seeds"].as_array().unwrap();
    let ctx = ExecutionContext::for_tests(0);
    for seed in seeds {
        let seed = seed.as_u64().unwrap();
        let mut workspace = MechanismWorkspace::default();
        let mut rng = CausalRng::from_seed(seed);
        let ancestral = interventional_mean(
            &model,
            &intervention,
            VariableId::from_raw(1),
            json_usize(&expected["acceptance"]["ancestral_draws"], "ancestral_draws"),
            &mut rng,
            &mut workspace,
            &ctx,
        )
        .unwrap();
        assert!(
            (ancestral - mean_target).abs()
                <= expected["acceptance"]["mean_atol"]["ancestral"].as_f64().unwrap()
        );

        let mut rng = CausalRng::from_seed(seed);
        let kde = KdeDoSampler::new(VariableId::from_raw(1))
            .sample(
                &model,
                &intervention,
                json_usize(&expected["acceptance"]["kde_draws"], "kde_draws"),
                &mut rng,
                &mut workspace,
                &ctx,
            )
            .unwrap();
        let (mean, variance) = moments(&kde.values);
        assert!(
            (mean - mean_target).abs()
                <= expected["acceptance"]["mean_atol"]["kde"].as_f64().unwrap()
        );
        assert!(
            (variance - variance_target).abs()
                <= expected["acceptance"]["variance_atol"]["kde"].as_f64().unwrap()
        );

        let mut rng = CausalRng::from_seed(seed);
        let mcmc = McmcDoSampler::new(VariableId::from_raw(1))
            .sample(
                &model,
                &intervention,
                json_usize(&expected["acceptance"]["mcmc_draws"], "mcmc_draws"),
                &mut rng,
                &mut workspace,
                &ctx,
            )
            .unwrap();
        let (mean, _) = moments(&mcmc.values);
        assert!(
            (mean - mean_target).abs()
                <= expected["acceptance"]["mean_atol"]["mcmc"].as_f64().unwrap(),
            "seed {seed}: MCMC mean={mean}"
        );
        let rate = mcmc.accept_rate.unwrap();
        let band = expected["acceptance"]["mcmc_accept_rate"].as_array().unwrap();
        assert!(rate >= band[0].as_f64().unwrap() && rate <= band[1].as_f64().unwrap());
    }
}
