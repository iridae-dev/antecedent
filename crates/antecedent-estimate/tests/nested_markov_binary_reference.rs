//! Constrained fit of the binary Verma pilot against a different optimizer.
//!
//! `oracle_data/nested_markov_binary_reference.py` (frozen in the JSON beside it) maximizes the
//! full eleven-parameter log-likelihood with SLSQP from twelve starts on eight count tables: a
//! table off the model at 1e6 and 1e12 total counts, tiny counts, nine decades of cell scale,
//! boxes near the parameter boundary, and random tables. The product (profiling `g` inside block
//! coordinate ascent over `q2`, `q4`) must reach the same maximum log-likelihood, never above it
//! beyond rounding, and the same model and plug-in do-means.

use antecedent_core::ExecutionContext;
use antecedent_estimate::nested_markov_binary::{
    BinaryCells, FitOptions, fit_nested_markov, log_likelihood, model_means, plugin_means,
};
use serde_json::Value as Json;

fn floats(value: &Json) -> Vec<f64> {
    value.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect()
}

#[test]
fn constrained_fit_reaches_the_reference_maximum_on_hard_tables() {
    let cases: Json =
        serde_json::from_str(include_str!("oracle_data/nested_markov_binary_reference.json"))
            .unwrap();
    let ctx = ExecutionContext::for_tests(3);
    let mut refused = 0;
    let mut fitted = 0;
    for case in cases["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let counts = floats(&case["counts"]);
        let total: f64 = counts.iter().sum();
        let table = BinaryCells::new(&counts).expect("counts");
        // A reference maximum within 1e-9 of a boundary is outside the interior model the
        // pilot fits: the product must refuse it as a boundary fit, never report it.
        let margin = case["min_fitted_margin"].as_f64().unwrap();
        let min_cell = case["min_fitted_cell"].as_f64().unwrap();
        if margin < 1e-9 || min_cell < 1e-9 {
            let refusal = fit_nested_markov(&table, &FitOptions::default(), &ctx)
                .expect_err(&format!("{name}: boundary maximum must refuse"));
            assert!(
                refusal.error.to_string().contains("nested_markov.fitted_boundary"),
                "{name}: {refusal:?}"
            );
            refused += 1;
            continue;
        }
        let fit = fit_nested_markov(&table, &FitOptions::default(), &ctx)
            .unwrap_or_else(|refusal| panic!("{name}: {refusal:?}"));
        fitted += 1;
        let reference = case["max_log_likelihood"].as_f64().unwrap();
        let ll = log_likelihood(table.counts(), &fit.cells);
        // never above the true maximum beyond rounding; the optimizer's maximum is reached
        let tolerance = 1e-9 * total.max(1.0);
        assert!(ll <= reference + tolerance, "{name}: product {ll} above reference {reference}");
        assert!(ll >= reference - tolerance, "{name}: product {ll} below reference {reference}");
        let saturated = case["saturated_log_likelihood"].as_f64().unwrap();
        let deviance = fit.diagnostics.deviance.unwrap();
        assert!((deviance - 2.0 * (saturated - ll)).abs() <= 1e-9 * total, "{name}: deviance");
        let model = model_means(&fit.parameters);
        let plugin = plugin_means(&table).unwrap();
        let ref_model = floats(&case["model_means"]);
        let ref_plugin = floats(&case["plugin_means"]);
        for k in 0..2 {
            assert!((plugin[k] - ref_plugin[k]).abs() <= 1e-12, "{name}: plug-in mean {k}");
            // the do-means are a function of the optimum; the likelihood is flat near it, so the
            // tolerance is the parameter accuracy a 1e-9 relative likelihood agreement allows
            assert!(
                (model[k] - ref_model[k]).abs() <= 1e-4,
                "{name}: model mean {k}: {} vs {}",
                model[k],
                ref_model[k]
            );
        }
    }
    assert_eq!(fitted + refused, 12);
    assert!(refused >= 1 && fitted >= 8, "{fitted} fitted, {refused} refused");
}
