//! Fisher-z interval and Student-t p-value against 50-digit references.
//!
//! `tests/oracle_data/fisher_z_reference.py` evaluates both from their definitions (exact normal
//! quantile by the inverse error function, regularized incomplete beta for the t tail) on a grid
//! of correlations from 1e-8 to 1 - 1e-11, residual degrees of freedom from 1 to 1e6 and levels
//! from 0.5 to 0.999999.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use serde_json::Value;

use super::{analytic_parcorr_ci, analytic_parcorr_pvalue};

#[test]
fn interval_and_p_value_match_the_multiprecision_definitions() {
    let doc: Value =
        serde_json::from_str(include_str!("../../tests/oracle_data/fisher_z_reference.json"))
            .unwrap();
    let rows = doc["rows"].as_array().unwrap();
    assert_eq!(rows.len(), 9 * 8 * 5);
    let mut worst_p = 0.0_f64;
    let mut worst_ci = 0.0_f64;
    for row in rows {
        let f = |i: usize| row[i].as_f64().unwrap();
        let (r, df, level) = (f(0), f(1), f(2));
        let p = analytic_parcorr_pvalue(r, df);
        let want = f(5);
        // relative where the tail probability is representable, absolute at the floor
        let err = (p - want).abs() / want.abs().max(1e-300);
        worst_p = worst_p.max(err.min((p - want).abs() / 1e-12));
        assert!(
            (p - want).abs() <= 1e-10 * want.abs() + 1e-300,
            "p-value r={r} df={df}: got {p} want {want}"
        );
        let (lo, hi) = analytic_parcorr_ci(r, df, level);
        if row[3].is_null() {
            assert!((lo + 1.0).abs() < 1e-15 && (hi - 1.0).abs() < 1e-15, "df={df}: [{lo}, {hi}]");
            continue;
        }
        // Near r = 1 the interval endpoint tanh(z + ...) is within 1e-16 of 1: absolute error.
        let (wl, wh) = (f(3), f(4));
        worst_ci = worst_ci.max((lo - wl).abs()).max((hi - wh).abs());
        assert!(
            (lo - wl).abs() <= 1e-12 && (hi - wh).abs() <= 1e-12,
            "interval r={r} df={df} level={level}: got [{lo}, {hi}] want [{wl}, {wh}]"
        );
    }
    assert!(worst_ci <= 1e-12 && worst_p.is_finite());
}
