//! The categorical regime under declared sampling weights, clusters and serial lags against an
//! exact rational reference.
//!
//! `oracle_data/categorical_declared_reference.py` (frozen in the JSON beside it) solves the
//! weighted normal equations in `Fraction` arithmetic and forms the sandwich of the weighted scores
//! for HC0 to HC3, the one-way cluster-robust meat (`G / (G - 1) (n - 1) / (n - p)`) and the
//! Newey-West meat, then the dummy covariance, the level and pairwise contrast variances and the
//! omnibus Wald statistic. The cases include unequal weights with two zero-weight rows and one
//! extreme weight, equal weights at a non-unit scale, an outcome offset of 1e8, unbalanced clusters
//! with singletons, weights composed with clusters, the smallest admissible clustered samples (three
//! clusters, and two clusters whose joint test does not exist) and serial lags 1 and 3.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_estimate::categorical_treatment::{
    CategoricalTreatmentInput, CategoricalTreatmentSpec, LevelScale,
    fit_categorical_treatment, fit_categorical_treatment_declared, fit_categorical_from_robust_summary,
    summarize_categorical_design_declared,
};
use antecedent_estimate::vector_treatment::{NamedColumn, RowStructure, VectorCovariance};
use serde_json::Value;

fn reference() -> Vec<Value> {
    serde_json::from_str::<Value>(include_str!("oracle_data/categorical_declared_reference.json"))
        .unwrap()
        .as_array()
        .cloned()
        .unwrap()
}

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn strings(v: &Value) -> Vec<String> {
    v.as_array().unwrap().iter().map(|s| s.as_str().unwrap().to_string()).collect()
}

fn close(got: f64, want: f64, rel: f64, what: &str) {
    assert!((got - want).abs() <= rel * want.abs().max(1e-300), "{what}: got {got:e} want {want:e}");
}

fn close_scaled(got: f64, want: f64, scale: f64, rel: f64, what: &str) {
    assert!((got - want).abs() <= rel * scale, "{what}: got {got:e} want {want:e}");
}

struct Case {
    input: CategoricalTreatmentInput<'static>,
    spec: CategoricalTreatmentSpec,
    weights: Option<Vec<f64>>,
    groups: Option<Vec<u32>>,
    lag: Option<usize>,
}

impl Case {
    fn new(case: &Value) -> Self {
        let adjustment = case["adjustment"]
            .as_array()
            .map(|_| vec![NamedColumn::new("g", floats(&case["adjustment"]))])
            .unwrap_or_default();
        let covariance = match case["covariance"].as_str().unwrap() {
            "hc0" => VectorCovariance::Hc0,
            "hc1" => VectorCovariance::Hc1,
            "hc2" => VectorCovariance::Hc2,
            "hc3" => VectorCovariance::Hc3,
            "cluster_robust" => VectorCovariance::Cluster,
            _ => VectorCovariance::NeweyWest,
        };
        let pairs = case["pairs"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| {
                let p = strings(p);
                (p[0].clone(), p[1].clone())
            })
            .collect();
        Self {
            input: CategoricalTreatmentInput {
                outcome: floats(&case["outcomes"]).into(),
                row_snapshot: "snap".into(),
                adjustment,
                levels: strings(&case["row_levels"]).into(),
            },
            spec: CategoricalTreatmentSpec {
                declared_levels: strings(&case["levels"]),
                scale: LevelScale::Unordered,
                reference: "a".into(),
                min_level_rows: 1,
                pairwise: pairs,
                monotonicity: None,
                covariance,
            },
            weights: case["weights"].is_array().then(|| floats(&case["weights"])),
            groups: case["groups"]
                .is_array()
                .then(|| floats(&case["groups"]).into_iter().map(|g| g as u32).collect()),
            lag: case["lag"].as_u64().map(|l| l as usize),
        }
    }

    fn rows(&self) -> RowStructure<'_> {
        RowStructure {
            sampling_weights: self.weights.as_deref(),
            clusters: self.groups.as_deref(),
            lag: self.lag,
        }
    }
}

fn tolerance(name: &str) -> f64 {
    if name.contains("offset") { 3e-6 } else { 1e-9 }
}

#[test]
fn weighted_clustered_and_serial_fits_match_the_exact_sandwich() {
    for case in reference() {
        let name = case["name"].as_str().unwrap();
        let want = &case["expected"];
        let c = Case::new(&case);
        let (fit, declared) = fit_categorical_treatment_declared(&c.input, &c.spec, &c.rows())
            .or_else(|error| {
                // The two-cluster joint test does not exist (the dummy covariance has rank one).
                assert!(want["omnibus"].is_null(), "{name}: {error}");
                Err(error)
            })
            .map_or_else(
                |error| {
                    assert!(error.to_string().contains("degenerate_covariance"), "{name}: {error}");
                    (None, None)
                },
                |(fit, declared)| (Some(fit), Some(declared)),
            );
        let (Some(fit), Some(declared)) = (fit, declared) else { continue };
        let tol = tolerance(name);
        let k = fit.coefficients.len();
        for (got, expected) in fit.coefficients.iter().zip(want["coefficients"].as_array().unwrap()) {
            assert_eq!(got.name, expected["name"].as_str().unwrap(), "{name}");
            close_scaled(got.estimate, expected["estimate"].as_f64().unwrap(), 1.0, tol, name);
        }
        let covariance = want["covariance"].as_array().unwrap();
        let scale = (0..k).map(|i| fit.covariance[i * k + i]).fold(0.0, f64::max);
        for a in 0..k {
            for b in 0..k {
                close_scaled(
                    fit.covariance[a * k + b],
                    covariance[a][b].as_f64().unwrap(),
                    scale,
                    tol,
                    &format!("{name} covariance[{a}][{b}]"),
                );
            }
        }
        for (got, expected) in fit.level_contrasts.iter().zip(want["levels"].as_array().unwrap()) {
            close_scaled(got.estimate, expected["estimate"].as_f64().unwrap(), 1.0, tol, name);
            close(
                got.standard_error.powi(2),
                expected["variance"].as_f64().unwrap(),
                tol.max(1e-9),
                &format!("{name} level variance"),
            );
        }
        for (got, expected) in fit.pairwise.iter().zip(want["pairs"].as_array().unwrap()) {
            close_scaled(got.estimate, expected["estimate"].as_f64().unwrap(), 1.0, tol, name);
            close(
                got.standard_error.powi(2),
                expected["variance"].as_f64().unwrap(),
                tol.max(1e-9),
                &format!("{name} pair variance"),
            );
        }
        if let Some(omnibus) = want["omnibus"].as_f64() {
            close(fit.omnibus.statistic, omnibus, tol.max(1e-8), &format!("{name} omnibus"));
        }
        for count in &fit.counts {
            assert_eq!(
                count.rows as u64,
                want["counts"][&count.level].as_u64().unwrap(),
                "{name} count of {}",
                count.level
            );
        }
        // The accounting of the declaration.
        assert_eq!(declared.clusters.map(|g| g as u64), want["clusters"].as_u64(), "{name}");
        assert_eq!(declared.lag, c.lag, "{name}");
        match declared.weights {
            Some(record) => {
                assert_eq!(record.zero_weight_rows, want["zero_weight_rows"].as_u64().unwrap(), "{name}");
                assert_eq!(record.rows_used, want["rows_used"].as_u64().unwrap(), "{name}");
                close(record.effective_sample_size, want["kish"].as_f64().unwrap(), 1e-12, name);
                close(record.weight_sum, want["weight_sum"].as_f64().unwrap(), 1e-12, name);
            }
            None => assert!(want["zero_weight_rows"].is_null(), "{name}"),
        }
    }
}

#[test]
fn the_robust_summary_of_declared_rows_replays_the_fit_without_a_row() {
    for case in reference() {
        let name = case["name"].as_str().unwrap();
        let c = Case::new(&case);
        let Ok((fit, _)) = fit_categorical_treatment_declared(&c.input, &c.spec, &c.rows()) else {
            continue;
        };
        let (summary, _) =
            summarize_categorical_design_declared(&c.input, &c.spec, &c.rows()).expect(name);
        let replay = fit_categorical_from_robust_summary(&c.spec, &summary).expect(name);
        assert_eq!(replay, fit, "{name}");
    }
}

#[test]
fn equal_weights_at_any_scale_are_the_unweighted_fit_and_plain_rows_are_the_historical_path() {
    let case = reference().into_iter().find(|c| c["name"] == "equal_weights_at_scale").unwrap();
    let c = Case::new(&case);
    let plain = fit_categorical_treatment(&c.input, &c.spec).unwrap();
    let (weighted, _) = fit_categorical_treatment_declared(&c.input, &c.spec, &c.rows()).unwrap();
    for (a, b) in plain.coefficients.iter().zip(&weighted.coefficients) {
        close(b.estimate, a.estimate, 1e-12, "estimate");
        close(b.standard_error, a.standard_error, 1e-12, "standard error");
    }
    let (declared_plain, record) =
        fit_categorical_treatment_declared(&c.input, &c.spec, &RowStructure::default()).unwrap();
    assert_eq!(declared_plain, plain);
    assert!(record.is_plain());
}

#[test]
fn a_zero_weight_row_is_a_deleted_row_and_a_level_of_zero_weight_rows_is_absent() {
    let case = reference().into_iter().find(|c| c["name"] == "weights_hc0_zero_and_extreme").unwrap();
    let c = Case::new(&case);
    let weights = c.weights.clone().unwrap();
    let keep: Vec<usize> = (0..weights.len()).filter(|&i| weights[i] > 0.0).collect();
    let pick = |values: &[f64]| keep.iter().map(|&i| values[i]).collect::<Vec<_>>();
    let deleted = CategoricalTreatmentInput {
        outcome: pick(&c.input.outcome).into(),
        row_snapshot: "snap".into(),
        adjustment: c.input.adjustment.iter().map(|a| NamedColumn::new("g", pick(&a.values))).collect(),
        levels: keep.iter().map(|&i| c.input.levels[i].clone()).collect::<Vec<_>>().into(),
    };
    let kept_weights = pick(&weights);
    let (with_zeros, _) = fit_categorical_treatment_declared(&c.input, &c.spec, &c.rows()).unwrap();
    let (without, _) = fit_categorical_treatment_declared(
        &deleted,
        &c.spec,
        &RowStructure { sampling_weights: Some(&kept_weights), ..RowStructure::default() },
    )
    .unwrap();
    assert_eq!(with_zeros, without);
    // Every row of level `c` given weight zero leaves the declared level absent: refused by name.
    let zeroed: Vec<f64> = weights
        .iter()
        .zip(c.input.levels.iter())
        .map(|(w, level)| if level == "c" { 0.0 } else { w.max(1.0) })
        .collect();
    let error = fit_categorical_treatment_declared(
        &c.input,
        &c.spec,
        &RowStructure { sampling_weights: Some(&zeroed), ..RowStructure::default() },
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("categorical_treatment.absent_level"), "{error}");
}

#[test]
fn declarations_without_their_covariance_kind_and_malformed_declarations_are_refused() {
    let case = reference().into_iter().find(|c| c["name"] == "weights_hc1").unwrap();
    let mut c = Case::new(&case);
    let n = c.input.outcome.len();
    let refuse = |c: &Case, rows: RowStructure<'_>| {
        fit_categorical_treatment_declared(&c.input, &c.spec, &rows).unwrap_err().to_string()
    };
    // Weights under the model-based variance.
    c.spec.covariance = VectorCovariance::ModelBased;
    let weights = vec![1.0; n];
    let text = refuse(&c, RowStructure { sampling_weights: Some(&weights), ..RowStructure::default() });
    assert!(text.contains("weights_need_robust_covariance"), "{text}");
    // Clusters under a covariance that does not read them, and the cluster kind without any.
    c.spec.covariance = VectorCovariance::Hc1;
    let groups: Vec<u32> = (0..n as u32).map(|i| i % 4).collect();
    let text = refuse(&c, RowStructure { clusters: Some(&groups), ..RowStructure::default() });
    assert!(text.contains("vector_treatment.cluster_unread"), "{text}");
    c.spec.covariance = VectorCovariance::Cluster;
    let text = refuse(&c, RowStructure::default());
    assert!(text.contains("vector_treatment.cluster_ids_required"), "{text}");
    let one = vec![3u32; n];
    let text = refuse(&c, RowStructure { clusters: Some(&one), ..RowStructure::default() });
    assert!(text.contains("vector_treatment.too_few_clusters"), "{text}");
    let short = vec![0u32; n - 1];
    let text = refuse(&c, RowStructure { clusters: Some(&short), ..RowStructure::default() });
    assert!(text.contains("categorical_treatment.row_count_mismatch"), "{text}");
    // Lags: required by the kind, unread by the others, never zero or capped.
    c.spec.covariance = VectorCovariance::Hc0;
    let text = refuse(&c, RowStructure { lag: Some(2), ..RowStructure::default() });
    assert!(text.contains("vector_treatment.lag_unread"), "{text}");
    c.spec.covariance = VectorCovariance::NeweyWest;
    let text = refuse(&c, RowStructure::default());
    assert!(text.contains("vector_treatment.lag_required"), "{text}");
    for lag in [0, n, n + 3] {
        let text = refuse(&c, RowStructure { lag: Some(lag), ..RowStructure::default() });
        assert!(text.contains("vector_treatment.invalid_lag"), "lag {lag}: {text}");
    }
    // Weights that are not a valid declaration.
    c.spec.covariance = VectorCovariance::Hc0;
    let mut bad = vec![1.0; n];
    bad[1] = f64::NAN;
    let text = refuse(&c, RowStructure { sampling_weights: Some(&bad), ..RowStructure::default() });
    assert!(text.contains("sampling_weights_invalid"), "{text}");
}
