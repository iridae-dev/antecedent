//! The dose-grid row under declared sampling weights, clusters and serial dependence against a
//! 50-digit reference.
//!
//! `oracle_data/dose_grid_declared_reference.py` (frozen in the JSON beside it) solves each
//! weighted Gaussian local quadratic in multiprecision arithmetic and forms the weighted scores,
//! the independent, one-way cluster-robust (`G / (G - 1)`) and Newey-West (Bartlett) standard
//! errors of the level, the derivative and a two-dose contrast, the Kish local effective size of
//! the weighted kernel, the worst-case smoothing bias, the folded-normal bias-aware interval and
//! the sup-t critical value of a two-dose band over the resampling units (rows, clusters or
//! blocks). The cases include unequal weights with two zero-weight rows and one extreme weight,
//! equal weights at a non-unit scale (which must equal the unweighted fit), unbalanced clusters
//! with singletons, weights composed with clusters, serial lags 1 and 4, and the smallest
//! admissible clustered sample (two clusters).

use antecedent_core::ExecutionContext;
use antecedent_estimate::dose_grid_functional::{
    ClaimSet, DoseDesign, DoseFunctional, DoseGridRequest, RowDeclaration, RowDependence,
    estimate_dose_grid_bias_aware_declared, estimate_dose_grid_functional,
    estimate_dose_grid_functional_declared,
};
use serde_json::Value;

const Z975: f64 = 1.959_963_984_540_054;

fn reference() -> Vec<Value> {
    serde_json::from_str::<Value>(include_str!("oracle_data/dose_grid_declared_reference.json"))
        .expect("frozen reference parses")
        .as_array()
        .cloned()
        .expect("array")
}

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().expect("array").iter().map(|x| x.as_f64().expect("number")).collect()
}

fn f(v: &Value, key: &str) -> f64 {
    v[key].as_f64().unwrap_or_else(|| panic!("{key}"))
}

fn close(got: f64, want: f64, rel: f64, what: &str) {
    assert!((got - want).abs() <= rel * want.abs().max(1e-300), "{what}: {got} vs {want}");
}

struct Case {
    dose: Vec<f64>,
    outcome: Vec<f64>,
    grid: Vec<f64>,
    weights: Option<Vec<f64>>,
    groups: Option<Vec<u32>>,
    lag: Option<usize>,
    bandwidth: f64,
}

impl Case {
    fn new(case: &Value) -> Self {
        Self {
            dose: floats(&case["dose"]),
            outcome: floats(&case["outcome"]),
            grid: floats(&case["grid"]),
            weights: case["weights"].is_array().then(|| floats(&case["weights"])),
            groups: case["groups"]
                .is_array()
                .then(|| floats(&case["groups"]).into_iter().map(|g| g as u32).collect()),
            lag: case["lag"].as_u64().map(|l| l as usize),
            bandwidth: f(case, "bandwidth"),
        }
    }

    fn request(&self, functional: DoseFunctional) -> DoseGridRequest<'_> {
        DoseGridRequest {
            dose: &self.dose,
            outcome: &self.outcome,
            grid: &self.grid,
            bandwidth: self.bandwidth,
            bandwidth_range: (self.bandwidth * 0.5, self.bandwidth * 2.0),
            minimum_local_ess: 3.0,
            functional,
            claims: ClaimSet { pointwise_level: true, derivative: true, simultaneous_band: false },
            design: DoseDesign::RandomizedDose,
        }
    }

    fn declaration(&self) -> RowDeclaration<'_> {
        RowDeclaration {
            sampling_weights: self.weights.as_deref(),
            dependence: match (&self.groups, self.lag) {
                (Some(groups), _) => RowDependence::Clustered(groups),
                (None, Some(lag)) => RowDependence::Serial { lag },
                (None, None) => RowDependence::Independent,
            },
        }
    }
}

fn quantile(clusters: Option<u64>) -> f64 {
    // The t_{G-1} multiplier of a clustered interval, from the shared few-cluster ratio.
    clusters.map_or(Z975, |g| Z975 * antecedent_stats::few_cluster_t_ratio(g as usize, 0.95))
}

#[test]
fn weighted_clustered_and_serial_levels_derivatives_and_support_match_the_multiprecision_fit() {
    let ctx = ExecutionContext::for_tests(3);
    for case in reference() {
        let name = case["name"].as_str().unwrap();
        let c = Case::new(&case);
        let declared = c.declaration();
        let points = case["points"].as_array().unwrap();
        let levels =
            estimate_dose_grid_functional_declared(&c.request(DoseFunctional::Level), &declared, &ctx)
                .expect(name);
        let slopes = estimate_dose_grid_functional_declared(
            &c.request(DoseFunctional::Derivative),
            &declared,
            &ctx,
        )
        .expect(name);
        assert_eq!(levels.n_rows as u64, case["rows_used"].as_u64().unwrap(), "{name}");
        for (j, p) in points.iter().enumerate() {
            let l = &levels.levels[j];
            close(l.level, f(p, "level"), 1e-8, name);
            close(l.standard_error, f(p, "level_se"), 1e-8, name);
            close(l.local_ess, f(p, "ess"), 1e-9, name);
            let q = quantile(case["clusters"].as_u64());
            close(l.interval.upper - l.level, q * f(p, "level_se"), 1e-8, name);
            let d = &slopes.derivatives[j];
            close(d.derivative, f(p, "derivative"), 1e-7, name);
            close(d.standard_error, f(p, "derivative_se"), 1e-8, name);
        }
        assert!(levels.numerical.max_abs_influence_sum < 1e-9, "{name}");
        // The accounting of the declaration is on the row.
        if c.weights.is_some() {
            let record = levels.declared.weights.expect("weights recorded");
            assert_eq!(record.zero_weight_rows, case["zero_weight_rows"].as_u64().unwrap(), "{name}");
            assert_eq!(record.rows_used, case["rows_used"].as_u64().unwrap(), "{name}");
            close(record.effective_sample_size, f(&case, "kish_weights"), 1e-9, name);
            close(record.weight_sum, f(&case, "weight_sum_used"), 1e-12, name);
        } else {
            assert!(levels.declared.weights.is_none(), "{name}");
        }
        assert_eq!(levels.declared.clusters.map(|g| g as u64), case["clusters"].as_u64(), "{name}");
        assert_eq!(levels.declared.lag, c.lag, "{name}");
    }
}

#[test]
fn weighted_clustered_and_serial_contrasts_match_the_multiprecision_difference_of_scores() {
    let ctx = ExecutionContext::for_tests(3);
    for case in reference() {
        let name = case["name"].as_str().unwrap();
        let Some(contrast) = case["contrast"].as_object() else { continue };
        let c = Case::new(&case);
        let functional = DoseFunctional::Contrast {
            from: contrast["from"].as_f64().unwrap(),
            to: contrast["to"].as_f64().unwrap(),
        };
        let row = estimate_dose_grid_functional_declared(&c.request(functional), &c.declaration(), &ctx)
            .expect(name);
        let got = row.contrast.expect("contrast");
        close(got.estimate, contrast["estimate"].as_f64().unwrap(), 1e-8, name);
        close(got.standard_error, contrast["se"].as_f64().unwrap(), 1e-8, name);
    }
}

#[test]
fn equal_weights_at_any_scale_reproduce_the_unweighted_row_bit_for_bit_in_the_estimate() {
    let ctx = ExecutionContext::for_tests(3);
    let case = reference().into_iter().find(|c| c["name"] == "equal_weights_scale_invariant").unwrap();
    let c = Case::new(&case);
    let plain = estimate_dose_grid_functional(&c.request(DoseFunctional::Level)).unwrap();
    let weighted = estimate_dose_grid_functional_declared(
        &c.request(DoseFunctional::Level),
        &c.declaration(),
        &ctx,
    )
    .unwrap();
    for (a, b) in plain.levels.iter().zip(&weighted.levels) {
        close(b.level, a.level, 1e-12, "level");
        close(b.standard_error, a.standard_error, 1e-12, "standard error");
        close(b.local_ess, a.local_ess, 1e-12, "ess");
    }
    // A plain declaration is the historical path itself.
    let plain_declared = estimate_dose_grid_functional_declared(
        &c.request(DoseFunctional::Level),
        &RowDeclaration::default(),
        &ctx,
    )
    .unwrap();
    assert_eq!(plain_declared, plain);
}

#[test]
fn a_zero_weight_row_is_exactly_a_deleted_row() {
    let ctx = ExecutionContext::for_tests(3);
    let case = reference().into_iter().find(|c| c["name"] == "weights_unequal_zero_and_extreme").unwrap();
    let c = Case::new(&case);
    let weights = c.weights.clone().unwrap();
    let keep: Vec<usize> = (0..c.dose.len()).filter(|&i| weights[i] > 0.0).collect();
    let dose: Vec<f64> = keep.iter().map(|&i| c.dose[i]).collect();
    let outcome: Vec<f64> = keep.iter().map(|&i| c.outcome[i]).collect();
    let kept_weights: Vec<f64> = keep.iter().map(|&i| weights[i]).collect();
    let mut deleted = c.request(DoseFunctional::Level);
    deleted.dose = &dose;
    deleted.outcome = &outcome;
    let with_zeros = estimate_dose_grid_functional_declared(
        &c.request(DoseFunctional::Level),
        &c.declaration(),
        &ctx,
    )
    .unwrap();
    let without = estimate_dose_grid_functional_declared(
        &deleted,
        &RowDeclaration { sampling_weights: Some(&kept_weights), dependence: RowDependence::Independent },
        &ctx,
    )
    .unwrap();
    assert_eq!(with_zeros.levels, without.levels);
    assert_eq!(with_zeros.declared.weights.unwrap().zero_weight_rows, 2);
}

#[test]
fn bias_aware_bounds_intervals_and_band_follow_the_declared_dependence() {
    let ctx = ExecutionContext::for_tests(11);
    for case in reference() {
        let name = case["name"].as_str().unwrap();
        let c = Case::new(&case);
        let declared = c.declaration();
        let m = f(&case, "curvature_bound");
        let points = case["points"].as_array().unwrap();
        for (functional, key) in
            [(DoseFunctional::Level, "level"), (DoseFunctional::Derivative, "derivative")]
        {
            let replicates = if c.grid.len() == 2 && key == "level" { 40_000 } else { 2_000 };
            let fit = estimate_dose_grid_bias_aware_declared(
                &c.request(functional),
                &declared,
                m,
                replicates,
                &ctx,
            )
            .expect(name);
            for (j, p) in points.iter().enumerate() {
                close(fit.smoothing_bias_bound[j], f(p, &format!("{key}_bias")), 1e-7, name);
                // The interval is the folded-normal bias-aware interval at the declared-dependence
                // standard error (the reference's `_half` is built from the same pair).
                let half = (fit.intervals[j].upper - fit.intervals[j].lower) / 2.0;
                close(half, f(p, &format!("{key}_half")), 1e-7, name);
            }
            if key == "level" && c.grid.len() == 2 {
                if let (Some(critical), Some(want)) =
                    (fit.band_critical_value, case["band_critical"].as_f64())
                {
                    // A multiplier critical value is a Monte Carlo rank: a few standard errors of
                    // the 0.95 rank at 40,000 replicates.
                    assert!((critical - want).abs() < 0.06, "{name}: {critical} vs {want}");
                }
            }
        }
        if let Some(contrast) = case["contrast"].as_object() {
            let functional = DoseFunctional::Contrast {
                from: contrast["from"].as_f64().unwrap(),
                to: contrast["to"].as_f64().unwrap(),
            };
            let fit = estimate_dose_grid_bias_aware_declared(
                &c.request(functional),
                &declared,
                m,
                0,
                &ctx,
            )
            .expect(name);
            close(fit.smoothing_bias_bound[0], contrast["bias_sum"].as_f64().unwrap(), 1e-7, name);
            let half = (fit.intervals[0].upper - fit.intervals[0].lower) / 2.0;
            close(half, contrast["half"].as_f64().unwrap(), 1e-7, name);
        }
    }
}

#[test]
fn declarations_the_rows_cannot_honour_are_refused_with_typed_details() {
    let ctx = ExecutionContext::for_tests(3);
    let case = reference().into_iter().find(|c| c["name"] == "serial_lag_1").unwrap();
    let c = Case::new(&case);
    let request = c.request(DoseFunctional::Level);
    let detail = |declared: RowDeclaration<'_>| {
        estimate_dose_grid_functional_declared(&request, &declared, &ctx)
            .expect_err("must refuse")
            .to_string()
    };
    let n = c.dose.len();
    // Lengths.
    let short = vec![1.0; n - 1];
    assert!(detail(RowDeclaration { sampling_weights: Some(&short), ..RowDeclaration::default() })
        .contains("dose_grid.invalid_request"));
    let few = vec![0u32; n - 1];
    assert!(detail(RowDeclaration {
        dependence: RowDependence::Clustered(&few),
        ..RowDeclaration::default()
    })
    .contains("dose_grid.invalid_request"));
    // Weights that are not a valid declaration.
    let mut bad = vec![1.0; n];
    bad[2] = -0.5;
    let text = detail(RowDeclaration { sampling_weights: Some(&bad), ..RowDeclaration::default() });
    assert!(text.contains("sampling_weights_invalid"), "{text}");
    let zeros = vec![0.0; n];
    assert!(
        detail(RowDeclaration { sampling_weights: Some(&zeros), ..RowDeclaration::default() })
            .contains("sampling_weights_invalid")
    );
    // One cluster is not a cluster-robust variance.
    let one = vec![7u32; n];
    assert!(detail(RowDeclaration {
        dependence: RowDependence::Clustered(&one),
        ..RowDeclaration::default()
    })
    .contains("dose_grid.too_few_clusters"));
    // The lag is never silently capped or zero.
    for lag in [0, n, n + 5] {
        let text = detail(RowDeclaration {
            dependence: RowDependence::Serial { lag },
            ..RowDeclaration::default()
        });
        assert!(text.contains("dose_grid.invalid_lag"), "lag {lag}: {text}");
    }
    // One weight that dwarfs the rest collapses the Kish size of its window: refused, not fitted.
    let mut dominant = vec![1.0; n];
    dominant[17] = 1.0e9;
    let mut at_row = c.request(DoseFunctional::Level);
    let spot = [c.dose[17]];
    at_row.grid = &spot;
    let error = estimate_dose_grid_functional_declared(
        &at_row,
        &RowDeclaration { sampling_weights: Some(&dominant), ..RowDeclaration::default() },
        &ctx,
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("dose_grid.insufficient_local_weight"), "{error}");
    // A weight that deletes every row in a window leaves the fit unsupported, never extrapolated.
    let mut tail = vec![1.0; n];
    for (i, w) in tail.iter_mut().enumerate() {
        if c.dose[i] > 1.0 {
            *w = 0.0;
        }
    }
    let mut far = c.request(DoseFunctional::Level);
    let grid = [2.5];
    far.grid = &grid;
    let error = estimate_dose_grid_functional_declared(
        &far,
        &RowDeclaration { sampling_weights: Some(&tail), ..RowDeclaration::default() },
        &ctx,
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("dose_grid.unsupported_dose"), "{error}");
}
