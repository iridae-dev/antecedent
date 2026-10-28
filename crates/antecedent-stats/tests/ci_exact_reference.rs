//! Partial-correlation and G-squared conditional-independence tests against 60-digit values.
//!
//! `oracle_data/ci_reference.py` writes `oracle_data/ci_reference.json`.  Partial correlation:
//! the reference residualizes in multiprecision arithmetic from the stored binary64 columns and
//! evaluates the Student-t p-value and Fisher-z interval from their definitions; the cases
//! include nearly collinear conditioners, a 1e8 offset, an absolute correlation within 5e-5 of
//! one, three residual degrees of freedom and n = 3000.  G-squared: the reference works from
//! integer contingency counts per stratum (no row-level data), and the test expands the counts
//! into columns, so the code's own contingency construction, stratification, degrees-of-freedom
//! rule and chi-square tail are all exercised.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::ExecutionContext;
use antecedent_stats::{
    CiBatchRequest, CiQuery, CiWorkspace, ConditionalIndependenceTest, ConfidenceMethod, GSquared,
    PartialCorrelation, SignificanceMethod, StatsError,
};
use serde_json::Value;

const REFERENCE: &str = include_str!("oracle_data/ci_reference.json");

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn rel(got: f64, want: f64) -> f64 {
    (got - want).abs() / want.abs().max(f64::MIN_POSITIVE)
}

fn run(
    test: &dyn ConditionalIndependenceTest,
    cols: &[&[f64]],
    z_flat: &[usize],
    confidence: ConfidenceMethod,
) -> Result<antecedent_stats::CiResult, StatsError> {
    let queries = [CiQuery { x: 0, y: 1, z_start: 0, z_len: z_flat.len() }];
    let req = CiBatchRequest {
        columns: cols,
        queries: &queries,
        z_flat,
        significance: SignificanceMethod::Analytic,
        confidence,
    };
    let mut ws = CiWorkspace::default();
    let mut out = test.test_batch_adhoc(&req, &mut ws, &ExecutionContext::for_tests(7))?;
    Ok(out.results.remove(0))
}

#[test]
fn partial_correlation_matches_multiprecision_residualization() {
    let doc: Value = serde_json::from_str(REFERENCE).unwrap();
    let mut failures = Vec::new();
    for case in doc["parcorr"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let owned: Vec<Vec<f64>> = case["columns"].as_array().unwrap().iter().map(floats).collect();
        let cols: Vec<&[f64]> = owned.iter().map(Vec::as_slice).collect();
        let z: Vec<usize> =
            case["z"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
        let want_r = case["r"].as_f64().unwrap();
        let want_p = case["p"].as_f64().unwrap();
        let want_ci = floats(&case["ci"]);
        let got =
            run(&PartialCorrelation::new(), &cols, &z, ConfidenceMethod::Analytic { level: 0.95 })
                .unwrap();
        let (lo, hi) = got.ci.expect("analytic interval requested");
        // The statistic is judged against the absolute scale 1 - |r| as well, because r near one
        // is determined to relative 1e-16 only in 1 - |r|.
        let r_err = (got.statistic - want_r).abs();
        let r_tol = 5e-13 * (1.0 - want_r.abs()).max(1e-300) + 4e-16;
        eprintln!(
            "{name:<30} df {} |dr| {:.1e} p rel {:.1e} ci rel {:.1e}/{:.1e}",
            got.df,
            r_err,
            rel(got.p_value, want_p),
            rel(lo, want_ci[0]),
            rel(hi, want_ci[1])
        );
        if r_err > r_tol {
            failures.push(format!("{name}: r {} vs {want_r} (tol {r_tol:e})", got.statistic));
        }
        if rel(got.p_value, want_p) > 1e-8 {
            failures.push(format!("{name}: p {:e} vs {want_p:e}", got.p_value));
        }
        // The interval endpoints are tanh of a shifted z; compare in absolute terms near one.
        for (g, w, side) in [(lo, want_ci[0], "lo"), (hi, want_ci[1], "hi")] {
            if (g - w).abs() > 1e-9 * (1.0 - w.abs()).max(1e-12) + 4e-16 {
                failures.push(format!("{name}: ci {side} {g} vs {w}"));
            }
        }
        if (got.df - case["df"].as_f64().unwrap()).abs() > 0.0 {
            failures.push(format!("{name}: df {} vs {}", got.df, case["df"]));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// Expand per-stratum contingency counts into (x, y, z) columns of integer codes.
fn expand(tables: &[Value]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let (mut x, mut y, mut z) = (Vec::new(), Vec::new(), Vec::new());
    for (s, table) in tables.iter().enumerate() {
        for (i, row) in table.as_array().unwrap().iter().enumerate() {
            for (j, count) in row.as_array().unwrap().iter().enumerate() {
                for _ in 0..count.as_u64().unwrap() {
                    x.push(i as f64);
                    y.push(j as f64);
                    z.push(s as f64);
                }
            }
        }
    }
    (x, y, z)
}

#[test]
fn g_squared_matches_exact_contingency_arithmetic() {
    let doc: Value = serde_json::from_str(REFERENCE).unwrap();
    let mut failures = Vec::new();
    for case in doc["gsquared"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let tables = case["tables"].as_array().unwrap();
        let (x, y, z) = expand(tables);
        let conditioned = tables.len() > 1;
        let (cols, z_flat): (Vec<&[f64]>, Vec<usize>) =
            if conditioned { (vec![&x, &y, &z], vec![2]) } else { (vec![&x, &y], vec![]) };
        let got = run(&GSquared::new(), &cols, &z_flat, ConfidenceMethod::None).unwrap();
        let (want_g, want_p) = (case["g"].as_f64().unwrap(), case["p"].as_f64().unwrap());
        eprintln!(
            "{name:<30} g rel {:.1e} p rel {:.1e} df {}",
            rel(got.statistic, want_g),
            rel(got.p_value, want_p),
            got.df
        );
        if (got.df - case["df"].as_f64().unwrap()).abs() > 0.0 {
            failures.push(format!("{name}: df {} vs {}", got.df, case["df"]));
        }
        if rel(got.statistic, want_g) > 1e-12 {
            failures.push(format!("{name}: g {} vs {want_g}", got.statistic));
        }
        if rel(got.p_value, want_p) > 1e-10 {
            failures.push(format!("{name}: p {:e} vs {want_p:e}", got.p_value));
        }
        // The same strata addressed through two conditioning columns (s mod 2, s div 2) must be
        // the same strata: the grouping key is the whole code vector, not a hash of it.
        if tables.len() >= 4 {
            let z_a: Vec<f64> = z.iter().map(|s| s % 2.0).collect();
            let z_b: Vec<f64> = z.iter().map(|s| (s / 2.0).floor()).collect();
            let two = run(&GSquared::new(), &[&x, &y, &z_a, &z_b], &[2, 3], ConfidenceMethod::None)
                .unwrap();
            if two.df != got.df || rel(two.statistic, got.statistic) > 1e-13 {
                failures.push(format!("{name}: two-column strata differ: {two:?} vs {got:?}"));
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// The refusals the record promises: a sparse table, and a table with no degrees of freedom,
/// are errors rather than a p-value.
#[test]
fn g_squared_refuses_sparse_and_degenerate_tables() {
    // 2x2 with a cell whose expected count is 2.5 (< 5).
    let (x, y, _) = expand(&[serde_json::json!([[2, 3], [4, 11]])]);
    let r = run(&GSquared::new(), &[&x, &y], &[], ConfidenceMethod::None);
    assert!(matches!(r, Err(StatsError::Unsupported { .. })), "{r:?}");
    // X constant in every stratum: zero degrees of freedom, so no chi-square reference.
    let x = vec![0.0; 40];
    let y: Vec<f64> = (0..40).map(|i| f64::from(i % 2)).collect();
    let r = run(&GSquared::new(), &[&x, &y], &[], ConfidenceMethod::None);
    assert!(matches!(r, Err(StatsError::Unsupported { .. })), "{r:?}");
}
