//! The dose-grid functional row against a 50-digit local quadratic reference.
//!
//! `oracle_data/dose_grid_local_quadratic_reference.py` (frozen in the JSON beside it) solves
//! each Gaussian-weighted quadratic fit in multiprecision arithmetic and forms the linear
//! smoother weights, the heteroskedasticity-robust sandwich standard errors of the level, the
//! derivative and a two-dose contrast, the Kish local effective sample size, the third-derivative
//! worst-case smoothing bias, the folded-normal bias-aware critical value and (for two-dose
//! grids) the sup-t critical value by quadrature of the bivariate normal. The cases include doses
//! offset by 1e8 and -4e6, a bandwidth 40 times the dose range, a window whose effective sample
//! size is near the support floor and an almost exactly quadratic mean.

use antecedent_core::{ExecutionContext, SupportStatus};
use antecedent_estimate::dose_grid_functional::{
    ClaimSet, DoseDesign, DoseFunctional, DoseGridRequest, dose_support_report,
    estimate_dose_grid_bias_aware, estimate_dose_grid_functional,
};
use serde_json::Value;

const Z975: f64 = 1.959_963_984_540_054;

fn reference() -> Vec<Value> {
    serde_json::from_str::<Value>(include_str!("oracle_data/dose_grid_local_quadratic_reference.json"))
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
    bandwidth: f64,
}

impl Case {
    fn new(case: &Value) -> Self {
        Self {
            dose: floats(&case["dose"]),
            outcome: floats(&case["outcome"]),
            grid: floats(&case["grid"]),
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
}

#[test]
fn level_derivative_standard_errors_and_support_match_the_multiprecision_fit() {
    for case in reference() {
        let name = case["name"].as_str().unwrap();
        let c = Case::new(&case);
        let points = case["points"].as_array().unwrap();
        let levels = estimate_dose_grid_functional(&c.request(DoseFunctional::Level)).expect(name);
        let slopes = estimate_dose_grid_functional(&c.request(DoseFunctional::Derivative)).expect(name);
        for (j, p) in points.iter().enumerate() {
            let l = &levels.levels[j];
            close(l.level, f(p, "level"), 1e-8, name);
            close(l.standard_error, f(p, "level_se"), 1e-8, name);
            close(l.local_ess, f(p, "ess"), 1e-10, name);
            close(l.interval.upper - l.level, Z975 * f(p, "level_se"), 1e-8, name);
            let d = &slopes.derivatives[j];
            close(d.derivative, f(p, "derivative"), 1e-7, name);
            close(d.standard_error, f(p, "derivative_se"), 1e-8, name);
            assert_eq!(l.support, SupportStatus::Supported, "{name}");
        }
        assert!(levels.numerical.max_abs_influence_sum < 1e-9, "{name}");
        // The support label flips at the Kish size: just below the reference it is weak overlap,
        // just above it is supported; a dose outside the observed range is outside the support.
        let ess = points.iter().map(|p| f(p, "ess")).collect::<Vec<_>>();
        for (j, e) in ess.iter().enumerate() {
            let at = [c.grid[j]];
            let label = |floor: f64| {
                dose_support_report(&c.dose, &at, c.bandwidth, floor).unwrap().point_status.unwrap()[0]
            };
            assert_eq!(label(e * (1.0 + 1e-9)), SupportStatus::WeakOverlap, "{name}");
            assert_eq!(label((e * (1.0 - 1e-9)).max(3.0)), SupportStatus::Supported, "{name}");
        }
        let top = c.dose.iter().copied().fold(f64::MIN, f64::max);
        let past = [top + 1e-6 * top.abs().max(1.0)];
        assert_eq!(
            dose_support_report(&c.dose, &past, c.bandwidth, 3.0).unwrap().point_status.unwrap()[0],
            SupportStatus::OutsideEmpiricalSupport,
            "{name}"
        );
    }
}

#[test]
fn contrast_estimate_and_covariance_aware_standard_error_match() {
    for case in reference() {
        let name = case["name"].as_str().unwrap();
        let Some(contrast) = case["contrast"].as_object() else { continue };
        let c = Case::new(&case);
        let functional = DoseFunctional::Contrast {
            from: contrast["from"].as_f64().unwrap(),
            to: contrast["to"].as_f64().unwrap(),
        };
        let row = estimate_dose_grid_functional(&c.request(functional)).expect(name);
        let got = row.contrast.expect("contrast");
        close(got.estimate, contrast["estimate"].as_f64().unwrap(), 1e-8, name);
        close(got.standard_error, contrast["se"].as_f64().unwrap(), 1e-8, name);
    }
}

#[test]
fn bias_aware_bounds_critical_values_and_band_match() {
    let ctx = ExecutionContext::for_tests(11);
    for case in reference() {
        let name = case["name"].as_str().unwrap();
        let c = Case::new(&case);
        let m = f(&case, "curvature_bound");
        let points = case["points"].as_array().unwrap();
        for (functional, key) in [(DoseFunctional::Level, "level"), (DoseFunctional::Derivative, "derivative")] {
            let replicates = if c.grid.len() == 2 && key == "level" { 40_000 } else { 2_000 };
            let fit = estimate_dose_grid_bias_aware(&c.request(functional), m, replicates, &ctx).expect(name);
            for (j, p) in points.iter().enumerate() {
                close(fit.smoothing_bias_bound[j], f(p, &format!("{key}_bias")), 1e-7, name);
                let half = (fit.intervals[j].upper - fit.intervals[j].lower) / 2.0;
                close(half, f(p, &format!("{key}_half")), 1e-7, name);
                assert!(fit.intervals[j].smoothing_bias_included, "{name}");
            }
            if let (Some(band), Some(critical)) = (&fit.band, fit.band_critical_value) {
                // A multiplier critical value is a Monte Carlo rank: tolerance is a few standard
                // errors of the 0.95 rank at 40,000 replicates.
                if key == "level" && c.grid.len() == 2 {
                    assert!((critical - f(&case, "band_critical")).abs() < 0.05, "{name}: {critical}");
                }
                for (j, p) in points.iter().enumerate() {
                    let est = f(p, key);
                    let se = f(p, &format!("{key}_se"));
                    let b = f(p, &format!("{key}_bias"));
                    close(band.upper[j], est + critical * se + b, 1e-7, name);
                    close(band.lower[j], est - critical * se - b, 1e-7, name);
                }
            }
        }
        if let Some(contrast) = case["contrast"].as_object() {
            let functional = DoseFunctional::Contrast {
                from: contrast["from"].as_f64().unwrap(),
                to: contrast["to"].as_f64().unwrap(),
            };
            let fit = estimate_dose_grid_bias_aware(&c.request(functional), m, 0, &ctx).expect(name);
            close(fit.smoothing_bias_bound[0], contrast["bias_sum"].as_f64().unwrap(), 1e-7, name);
            let half = (fit.intervals[0].upper - fit.intervals[0].lower) / 2.0;
            close(half, contrast["half"].as_f64().unwrap(), 1e-7, name);
        }
    }
}

#[test]
fn a_window_with_enough_kish_size_but_weight_sum_below_three_is_labelled_supported_and_refused_at_the_fit() {
    // Eight doses at |u| = 1.00 to 1.03 around the evaluation dose 0 with a bandwidth that gives each
    // a kernel weight of 0.3: the Kish size is 8 (above the floor of 3) while the weight sum is
    // 2.4, which the local quadratic refuses because its residual scale is not estimable.
    let h = 1.0 / (-2.0_f64 * 0.3_f64.ln()).sqrt();
    let dose = [-1.03, -1.02, -1.01, -1.0, 1.0, 1.01, 1.02, 1.03];
    let outcome = [0.2, -0.1, 0.3, 0.0, 0.1, -0.2, 0.25, 0.05];
    let report = dose_support_report(&dose, &[0.0], h, 3.0).expect("report");
    assert_eq!(report.point_status.unwrap()[0], SupportStatus::Supported);
    let request = DoseGridRequest {
        dose: &dose,
        outcome: &outcome,
        grid: &[0.0],
        bandwidth: h,
        bandwidth_range: (h * 0.5, h * 2.0),
        minimum_local_ess: 3.0,
        functional: DoseFunctional::Level,
        claims: ClaimSet { pointwise_level: true, derivative: true, simultaneous_band: false },
        design: DoseDesign::RandomizedDose,
    };
    let error = estimate_dose_grid_functional(&request).unwrap_err().to_string();
    assert!(error.contains("insufficient_local_weight"), "{error}");
}
