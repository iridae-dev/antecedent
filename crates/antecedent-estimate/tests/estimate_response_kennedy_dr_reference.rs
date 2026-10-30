//! Kennedy-style cross-fitted pseudo-outcome and curve against an exact rational pipeline.
//!
//! `oracle_data/estimate_response_kennedy_dr_reference.py` (frozen in the JSON beside it) follows the
//! estimator's definition on dyadic data: index-modulo folds, the penalized additive spline
//! nuisances of the outcome and of the treatment on the training rows at their unique exact
//! solution, the treatment scale with the effective-degrees-of-freedom divisor, the conditional and
//! the training-mixture marginal densities, the dimensionless floor on sigma times the conditional density, the
//! marginalized outcome over every row, the pseudo-outcome, the covariate term, and the
//! Gaussian-kernel local-quadratic level. The product uses a certified direct solve where available, otherwise a backfit; the
//! comparison keeps its original tolerance rather than claiming rounding accuracy. The
//! influence the product exports must equal the separately reviewed local-quadratic influence of
//! its own pseudo-outcome plus the reference covariate term (c_i - mean c) / n.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    AssumptionSet, ContinuousDomain, GridSpec, IdentificationStatus, ResponseFunctional,
    ResponseIdentification, ResponseQuery, ResponseValue, VariableId,
};
use antecedent_data::TabularData;
use antecedent_estimate::response::ContinuousResponseEstimator;
use antecedent_stats::{
    LocalQuadraticWorkspace, gaussian_local_quadratic_adjusted_influence_prechecked,
};
use serde_json::Value;

fn floats(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_str().map_or_else(|| x.as_f64().unwrap(), |s| s.parse().unwrap()))
        .collect()
}

fn diagnostic(response: &antecedent_core::CausalResponse, id: &str) -> Vec<f64> {
    response
        .support
        .diagnostics
        .iter()
        .find(|d| d.id.as_ref() == id)
        .unwrap_or_else(|| panic!("missing diagnostic {id}"))
        .values
        .to_vec()
}

#[test]
fn the_pseudo_outcome_level_and_influence_match_the_exact_pipeline() {
    let frozen: Value = serde_json::from_str(include_str!(
        "oracle_data/estimate_response_kennedy_dr_reference.json"
    ))
    .unwrap();
    for case in frozen["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let (a, x, y) = (floats(&case["a"]), floats(&case["x"]), floats(&case["y"]));
        let n = a.len();
        let data = TabularData::from_f64_columns([
            ("a", a.as_slice()),
            ("y", y.as_slice()),
            ("x", x.as_slice()),
        ])
        .unwrap();
        let (a_id, y_id, x_id) =
            (VariableId::from_raw(0), VariableId::from_raw(1), VariableId::from_raw(2));
        let grid = floats(&case["grid"]);
        let bandwidth = case["bandwidth"].as_f64().unwrap();
        let mut estimator = ContinuousResponseEstimator::new([x_id]);
        estimator.options.folds = usize::try_from(case["folds"].as_u64().unwrap()).unwrap();
        estimator.options.nuisance_basis =
            usize::try_from(case["basis"].as_u64().unwrap()).unwrap();
        estimator.options.nuisance_lambda = case["lambda"].as_f64().unwrap();
        estimator.options.bandwidth = Some(bandwidth);
        estimator.options.export_row_diagnostics = true;
        let query = ResponseQuery::new(ResponseFunctional::MeanCurve {
            outcome: y_id,
            treatment: ContinuousDomain::new(a_id, GridSpec::Values(Arc::from(grid.clone()))),
        });
        if name == "treatment_nearly_determined" {
            let pivots = floats(&case["outcome_min_ldl_pivots"]);
            assert_eq!(pivots.len(), 4);
            assert!(pivots.iter().all(|pivot| *pivot > 0.0));
        }
        let response = estimator
            .estimate_identified(
                &data,
                &query,
                IdentificationStatus::NonparametricallyIdentified,
                AssumptionSet::new(),
            )
            .unwrap_or_else(|e| panic!("{name}: {e}"));

        // The pseudo-outcome row by row, to the backfit tolerance of the nuisance fits.
        let pseudo = diagnostic(&response, "response.row_pseudo_outcome");
        let want = floats(&case["pseudo"]);
        let spread = y.iter().copied().fold(f64::NEG_INFINITY, f64::max)
            - y.iter().copied().fold(f64::INFINITY, f64::min);
        // The backfit stops on a change relative to the largest fitted value, intercept included,
        // so an outcome offset of 1e6 admits an absolute 1e-4 in the smooths (see the record).
        let magnitude = y.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        let tolerance = 2e-5 * spread.max(1.0) + 2e-9 * magnitude;
        if std::env::var("KDR_DEBUG").is_ok() {
            for (i, (g, w)) in pseudo.iter().zip(&want).enumerate() {
                eprintln!("{name} row {i}: {g} {w} {}", g - w);
            }
        }
        let worst = pseudo.iter().zip(&want).map(|(g, w)| (g - w).abs()).fold(0.0, f64::max);
        assert!(
            worst <= tolerance,
            "{name}: pseudo-outcome off by {worst} (tolerance {tolerance})"
        );

        // The curve is the local-quadratic level of the pseudo-outcome at each grid point.
        let ResponseIdentification::PointIdentified(ResponseValue::Surface { mean, .. }) =
            &response.estimate
        else {
            panic!("{name}: a mean curve is a surface");
        };
        for (g, (got, want)) in mean.iter().zip(floats(&case["level"])).enumerate() {
            assert!((got - want).abs() <= tolerance, "{name} grid {g}: level {got} vs {want}");
        }

        // The exported influence is the local-quadratic influence of the pseudo-outcome plus the
        // covariate term of the reference.
        let covariate = floats(&case["covariate_centered"]);
        let centre = covariate.iter().sum::<f64>() / n as f64;
        let influence = diagnostic(&response, "response.row_influence");
        let mut workspace = LocalQuadraticWorkspace::default();
        for (g, &at) in grid.iter().enumerate() {
            let local = gaussian_local_quadratic_adjusted_influence_prechecked(
                &mut workspace,
                &a,
                &pseudo,
                at,
                bandwidth,
            )
            .unwrap();
            for i in 0..n {
                let expected = local.influences[i] + (covariate[i] - centre) / n as f64;
                let got = influence[g * n + i];
                assert!(
                    (got - expected).abs() <= tolerance / n as f64 + 1e-12,
                    "{name} grid {g} row {i}: influence {got} vs {expected}"
                );
            }
        }
    }
}
