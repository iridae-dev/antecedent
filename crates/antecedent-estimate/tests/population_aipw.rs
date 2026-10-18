//! Independent point identities for explicit Gaussian population AIPW intervals.

use std::sync::Arc;

use antecedent_core::{
    AssumptionSet, AverageEffectQuery, CausalSchemaBuilder, ExecutionContext, MeasurementSpec,
    RoleHint, SmallRoleSet, ValueType, VariableId,
};
use antecedent_data::{
    Float64Column, OwnedColumn, OwnedColumnarStorage, TabularData, ValidityBitmap,
};
use antecedent_estimate::{AipwAte, AipwWorkspace, DmlAte, DrLearner, SamplingWeights};
use antecedent_expr::{ExprId, IdentifiedEstimand};

const T: VariableId = VariableId::from_raw(0);
const Y: VariableId = VariableId::from_raw(1);
const Z: VariableId = VariableId::from_raw(2);

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(19)
}

fn dataset(t: &[f64], y: &[f64], z: &[f64]) -> (TabularData, IdentifiedEstimand) {
    let n = t.len();
    let mut builder = CausalSchemaBuilder::new();
    for (name, hint) in [
        ("t", RoleHint::TreatmentCandidate),
        ("y", RoleHint::OutcomeCandidate),
        ("z", RoleHint::Context),
    ] {
        builder
            .add_variable(
                name,
                ValueType::Continuous,
                SmallRoleSet::from_hint(hint),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
    }
    let columns = [(T, t), (Y, y), (Z, z)]
        .into_iter()
        .map(|(id, values)| {
            OwnedColumn::Float64(
                Float64Column::new(id, Arc::from(values), ValidityBitmap::all_valid(n)).unwrap(),
            )
        })
        .collect();
    let storage =
        OwnedColumnarStorage::try_new(builder.build().unwrap(), columns, None, None).unwrap();
    let estimand =
        IdentifiedEstimand::backdoor("backdoor.adjustment", Arc::from([Z]), ExprId::from_raw(0));
    (TabularData::new(storage), estimand)
}

fn query() -> AverageEffectQuery {
    AverageEffectQuery::binary_ate(T, Y)
}

#[test]
fn arm_linear_target_is_reproduced_for_ols_and_corrected_ridge_without_changing_default() {
    let n = 120;
    let z: Vec<f64> = (0..n).map(|i| (i as f64 * 0.71).sin() + i as f64 / 120.0).collect();
    let t: Vec<f64> = (0..n).map(|i| (i % 2) as f64).collect();
    let y: Vec<f64> = (0..n).map(|i| z[i] + (1.0 + 0.5 * z[i]) * t[i]).collect();
    let w: Vec<f64> = (0..n).map(|i| 0.5 + (i % 7) as f64 / 4.0).collect();
    let truth =
        w.iter().zip(&z).map(|(w, z)| w * (1.0 + 0.5 * z)).sum::<f64>() / w.iter().sum::<f64>();
    let (data, estimand) = dataset(&t, &y, &z);
    let aipw = AipwAte::new()
        .with_bootstrap_replicates(0)
        .with_sampling_weights(SamplingWeights::new(w.clone()).unwrap());
    let p = aipw.prepare(&data, &estimand, &query()).unwrap();
    let original =
        aipw.fit(&p, &mut AipwWorkspace::default(), &ctx(), AssumptionSet::new()).unwrap();
    let interval = aipw
        .population_linear_interval(
            &p,
            &mut AipwWorkspace::default(),
            &ctx(),
            AssumptionSet::new(),
            0.95,
        )
        .unwrap();
    assert!((interval.estimate.ate - truth).abs() < 1e-10);
    assert!(interval.point_correction.abs() < 1e-10);
    assert!(interval.noise_variance < 1e-20);
    assert!(interval.target_variance > 0.0);
    assert_eq!(
        original.ate.to_bits(),
        aipw.fit(&p, &mut AipwWorkspace::default(), &ctx(), AssumptionSet::new())
            .unwrap()
            .ate
            .to_bits()
    );
    let dml = DmlAte::new().with_sampling_weights(SamplingWeights::new(w.clone()).unwrap());
    let p = dml.prepare(&data, &estimand, &query()).unwrap();
    let raw = dml.fit(&p, &ctx(), AssumptionSet::new()).unwrap();
    let dml_interval =
        dml.population_linear_interval(&p, &ctx(), AssumptionSet::new(), 0.95).unwrap();
    assert!((dml_interval.estimate.ate - truth).abs() < 1e-10);
    assert!((dml_interval.point_correction - (truth - raw.ate)).abs() < 1e-10);
    assert_eq!(raw.ate.to_bits(), dml.fit(&p, &ctx(), AssumptionSet::new()).unwrap().ate.to_bits());
    let dr = DrLearner::new().with_sampling_weights(SamplingWeights::new(w).unwrap());
    let p = dr.prepare(&data, &estimand, &query()).unwrap();
    let dr_interval =
        dr.population_linear_interval(&p, &ctx(), AssumptionSet::new(), 0.95).unwrap();
    assert!((dr_interval.estimate.ate - truth).abs() < 1e-10);
    assert!((dr_interval.estimate.ate - dml_interval.estimate.ate).abs() < 1e-10);
    assert!((dr_interval.lower - dml_interval.lower).abs() < 1e-10);
    assert!((dr_interval.upper - dml_interval.upper).abs() < 1e-10);
}

fn solve_two(g: [f64; 3], rhs: [f64; 2]) -> [f64; 2] {
    let det = g[0] * g[2] - g[1] * g[1];
    [(g[2] * rhs[0] - g[1] * rhs[1]) / det, (g[0] * rhs[1] - g[1] * rhs[0]) / det]
}

#[test]
#[allow(
    clippy::too_many_lines,
    reason = "independent logistic, training-map and HC3 oracles remain explicit together"
)]
fn atc_hc3_matches_independent_full_training_coefficient_and_heteroskedastic_oracle() {
    let n = 160;
    let z: Vec<f64> = (0..n).map(|i| (i as f64 * 0.71).sin() + i as f64 / 160.0).collect();
    let t: Vec<f64> = (0..n).map(|i| (i % 2) as f64).collect();
    let y: Vec<f64> =
        (0..n).map(|i| z[i] + t[i] + (0.5 + 0.5 * z[i].abs()) * (i as f64 * 2.13).sin()).collect();
    let mut w: Vec<f64> = (0..n).map(|i| 0.5 + (i % 7) as f64 / 4.0).collect();
    let scale = n as f64 / w.iter().sum::<f64>();
    for w in &mut w {
        *w *= scale;
    }
    let (data, estimand) = dataset(&t, &y, &z);
    let estimator = AipwAte::new()
        .with_bootstrap_replicates(0)
        .with_sampling_weights(SamplingWeights::new(w.clone()).unwrap());
    let p = estimator
        .prepare(
            &data,
            &estimand,
            &query().with_target_population(antecedent_core::TargetPopulation::Untreated),
        )
        .unwrap();
    let raw =
        estimator.fit(&p, &mut AipwWorkspace::default(), &ctx(), AssumptionSet::new()).unwrap();
    let result = estimator
        .population_linear_atc_interval(
            &p,
            &mut AipwWorkspace::default(),
            &ctx(),
            AssumptionSet::new(),
            0.95,
        )
        .unwrap();
    let mut beta = [0.0; 2];
    for _ in 0..30 {
        let mut g = [0.0; 3];
        let mut rhs = [0.0; 2];
        for i in 0..n {
            let e = 1.0 / (1.0 + (-beta[0] - beta[1] * z[i]).exp());
            let v = w[i] * e * (1.0 - e);
            g[0] += v;
            g[1] += v * z[i];
            g[2] += v * z[i] * z[i];
            rhs[0] += w[i] * (t[i] - e);
            rhs[1] += w[i] * (t[i] - e) * z[i];
        }
        let step = solve_two(g, rhs);
        beta[0] += step[0];
        beta[1] += step[1];
        if step[0].abs().max(step[1].abs()) < 1e-13 {
            break;
        }
    }
    let e: Vec<f64> = z
        .iter()
        .map(|z| (1.0 / (1.0 + (-beta[0] - beta[1] * z).exp())).clamp(0.01, 0.99))
        .collect();
    let controls = (0..n).filter(|&i| t[i] < 0.5).map(|i| w[i]).sum::<f64>();
    let mut g = [0.0; 3];
    let mut load = [0.0; 2];
    let mut h = vec![0.0; n];
    for i in 0..n {
        let r = if t[i] > 0.5 { (1.0 - e[i]) / e[i] } else { -1.0 };
        h[i] = w[i] / controls * r;
        load[0] -= h[i];
        load[1] -= h[i] * z[i];
        if t[i] > 0.5 {
            g[0] += w[i];
            g[1] += w[i] * z[i];
            g[2] += w[i] * z[i] * z[i];
        }
    }
    let b = solve_two(g, load);
    for i in 0..n {
        if t[i] > 0.5 {
            h[i] += w[i] * (b[0] + b[1] * z[i]);
        }
    }
    let mut variance = 0.0;
    let mut contrast = vec![0.0; n];
    for arm in [0.0, 1.0] {
        let mut g = [0.0; 3];
        let mut rhs = [0.0; 2];
        for i in 0..n {
            if (t[i] - arm).abs() < 0.1 {
                g[0] += 1.0;
                g[1] += z[i];
                g[2] += z[i] * z[i];
                rhs[0] += y[i];
                rhs[1] += z[i] * y[i];
            }
        }
        let b = solve_two(g, rhs);
        for i in 0..n {
            let mu = b[0] + b[1] * z[i];
            contrast[i] += if arm > 0.5 { mu } else { -mu };
            if (t[i] - arm).abs() < 0.1 {
                let projected = solve_two(g, [1.0, z[i]]);
                let leverage = projected[0] + z[i] * projected[1];
                variance += (h[i] * (y[i] - mu) / (1.0 - leverage)).powi(2);
            }
        }
    }
    let mean = (0..n).map(|i| w[i] * (1.0 - t[i]) * contrast[i]).sum::<f64>() / controls;
    let target = (0..n)
        .map(|i| (w[i] * (1.0 - t[i]) / controls * (contrast[i] - mean)).powi(2))
        .sum::<f64>()
        * n as f64
        / (n - 1) as f64;
    let point = h.iter().zip(&y).map(|(h, y)| h * y).sum::<f64>();
    assert!((point - raw.ate).abs() < 1e-9);
    assert_eq!(result.estimate.ate.to_bits(), raw.ate.to_bits());
    assert!((result.noise_variance - variance).abs() < 1e-9);
    assert!((result.target_variance - target).abs() < 1e-9);
    let radius = 1.959_963_984_540_054 * (variance + target).sqrt();
    assert!((result.lower - (point - radius)).abs() < 1e-9);
    assert!((result.upper - (point + radius)).abs() < 1e-9);
}
