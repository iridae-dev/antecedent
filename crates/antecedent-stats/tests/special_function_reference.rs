//! Special-function implementations against 50-digit `mpmath` values.
//!
//! The frozen values come from `oracle_data/special_functions_reference.py` (its output is
//! `oracle_data/special_functions_reference.json`).  Each argument is an exact binary64 value
//! and each expected value is computed from the mathematical definition (erfc, normal CDF,
//! regularized incomplete beta and gamma, log-gamma) and rounded once.  Quantiles are solved
//! on the high-precision CDF, never by inverting an approximation.  The cases deliberately
//! include the worst regimes for these routines: approximation-region boundaries, deep tails,
//! subnormal results, near-median Student-t, very large and very small degrees of freedom,
//! and incomplete-gamma arguments in the sqrt(a) transition region at large a.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_kernels::{erf, erfc, log_norm_cdf, norm_cdf, norm_inv, norm_sf};
use antecedent_stats::{chi_square_sf, gamma_q, ln_gamma, student_t_ppf, student_t_sf};
use serde_json::Value;

const REFERENCE: &str = include_str!("oracle_data/special_functions_reference.json");

fn rows(doc: &Value, name: &str) -> Vec<Vec<f64>> {
    doc[name]
        .as_array()
        .unwrap_or_else(|| panic!("missing {name}"))
        .iter()
        .map(|row| row.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect())
        .collect()
}

/// Relative error that treats subnormal expected values at the spacing of the format.
fn rel(got: f64, want: f64) -> f64 {
    if want == 0.0 {
        return got.abs();
    }
    let scale = want.abs().max(f64::MIN_POSITIVE);
    (got - want).abs() / scale
}

struct Sweep {
    name: &'static str,
    tol: f64,
    worst: f64,
    worst_at: String,
    failures: Vec<String>,
}

impl Sweep {
    fn new(name: &'static str, tol: f64) -> Self {
        Self { name, tol, worst: 0.0, worst_at: String::new(), failures: Vec::new() }
    }

    fn check(&mut self, at: String, got: f64, want: f64) {
        let e = if got.is_finite() { rel(got, want) } else { f64::INFINITY };
        if e > self.worst {
            self.worst = e;
            self.worst_at = at.clone();
        }
        if e > self.tol {
            self.failures.push(format!("{at}: got {got:e} want {want:e} rel {e:e}"));
        }
    }

    /// Report the worst case; the failures are appended to `sink` so one run lists every
    /// function that is out of tolerance rather than stopping at the first.
    fn finish(self, sink: &mut Vec<String>) {
        eprintln!("{:<14} worst relative error {:.2e} at {}", self.name, self.worst, self.worst_at);
        if !self.failures.is_empty() {
            sink.push(format!(
                "{} outside tolerance {:e}:\n{}",
                self.name,
                self.tol,
                self.failures.join("\n")
            ));
        }
    }
}

#[test]
fn kernel_normal_family_matches_high_precision() {
    let doc: Value = serde_json::from_str(REFERENCE).unwrap();
    let mut sink = Vec::new();
    let mut s = Sweep::new("erf", 4e-16);
    for r in rows(&doc, "erf") {
        s.check(format!("erf({:e})", r[0]), erf(r[0]), r[1]);
    }
    s.finish(&mut sink);
    // erfc is a rational approximation whose result for negative arguments is 2 - mag; the
    // subnormal band (x > 26.5) can only be accurate to the absolute spacing of the format.
    let mut s = Sweep::new("erfc", 1e-15);
    for r in rows(&doc, "erfc") {
        if r[1] < 1e-300 && r[1] > 0.0 && r[0] > 0.0 {
            continue; // covered by the subnormal assertion below
        }
        s.check(format!("erfc({:e})", r[0]), erfc(r[0]), r[1]);
    }
    s.finish(&mut sink);
    for r in rows(&doc, "erfc") {
        if r[0] > 0.0 && r[1] < 1e-300 {
            let spacing = f64::from_bits(1);
            assert!((erfc(r[0]) - r[1]).abs() <= 8.0 * spacing + 1e-12 * r[1], "erfc({})", r[0]);
        }
    }
    let mut s = Sweep::new("norm_cdf", 1.5e-15);
    for r in rows(&doc, "norm_cdf") {
        s.check(format!("norm_cdf({:e})", r[0]), norm_cdf(r[0]), r[1]);
    }
    s.finish(&mut sink);
    let mut s = Sweep::new("norm_sf", 1.5e-15);
    for r in rows(&doc, "norm_sf") {
        if r[1] < 1e-300 {
            continue;
        }
        s.check(format!("norm_sf({:e})", r[0]), norm_sf(r[0]), r[1]);
    }
    s.finish(&mut sink);
    let mut s = Sweep::new("log_norm_cdf", 1e-15);
    for r in rows(&doc, "log_norm_cdf") {
        s.check(format!("log_norm_cdf({:e})", r[0]), log_norm_cdf(r[0]), r[1]);
    }
    s.finish(&mut sink);
    let mut s = Sweep::new("norm_inv", 1e-15);
    for r in rows(&doc, "norm_inv") {
        if r[0] == 0.5 {
            assert_eq!(norm_inv(0.5), 0.0);
            continue;
        }
        s.check(format!("norm_inv({:e})", r[0]), norm_inv(r[0]), r[1]);
    }
    s.finish(&mut sink);
    assert!(sink.is_empty(), "{}", sink.join("\n"));
}

#[test]
fn student_t_survival_and_quantile_match_high_precision() {
    let doc: Value = serde_json::from_str(REFERENCE).unwrap();
    let mut sink = Vec::new();
    let mut s = Sweep::new("student_t_sf", 2e-13);
    for r in rows(&doc, "student_t_sf") {
        s.check(
            format!("student_t_sf(t={:e}, df={:e})", r[0], r[1]),
            student_t_sf(r[0], r[1]),
            r[2],
        );
    }
    s.finish(&mut sink);
    let mut s = Sweep::new("student_t_ppf", 2e-13);
    for r in rows(&doc, "student_t_ppf") {
        s.check(
            format!("student_t_ppf(p={:e}, df={:e})", r[0], r[1]),
            student_t_ppf(r[0], r[1]),
            r[2],
        );
    }
    s.finish(&mut sink);
    assert!(sink.is_empty(), "{}", sink.join("\n"));
}

#[test]
fn incomplete_gamma_and_log_gamma_match_high_precision() {
    let doc: Value = serde_json::from_str(REFERENCE).unwrap();
    let mut sink = Vec::new();
    let mut s = Sweep::new("gamma_q", 5e-13);
    for r in rows(&doc, "gamma_q") {
        s.check(format!("gamma_q(a={:e}, x={:e})", r[0], r[1]), gamma_q(r[0], r[1]), r[2]);
    }
    s.finish(&mut sink);
    // Chi-square is Q(df/2, x/2) at integer df.
    for df in [1usize, 2, 3, 7, 30] {
        for x in [0.5_f64, 3.0, 12.0, 80.0] {
            assert!((chi_square_sf(x, df) - gamma_q(df as f64 / 2.0, x / 2.0)).abs() == 0.0);
        }
    }
    let mut s = Sweep::new("ln_gamma", 1e-13);
    for r in rows(&doc, "ln_gamma") {
        // Lanczos error is absolute, so scale by max(|value|, 1) rather than by a near-zero value.
        let got = ln_gamma(r[0]);
        let e = (got - r[1]).abs() / r[1].abs().max(1.0);
        if e > 1e-14 {
            s.failures.push(format!("ln_gamma({:e}): got {got:e} want {:e}", r[0], r[1]));
        }
        s.worst = s.worst.max(e);
    }
    s.finish(&mut sink);
    assert!(sink.is_empty(), "{}", sink.join("\n"));
}

/// Behaviour at the edges of the domain that the grid above cannot express.
#[test]
fn student_t_edges_do_not_lose_the_answer() {
    // Squaring |t| > 1e154 used to overflow and turn a valid (underflowing) tail into NaN.
    assert!(student_t_sf(1e200, 3.0) >= 0.0);
    assert!(student_t_sf(-1e200, 3.0) <= 1.0 && !student_t_sf(-1e200, 3.0).is_nan());
    // Heavy tails: the bracket must grow to the root instead of returning its old 1e15 cap.
    let q = student_t_ppf(1e-10, 0.3);
    assert!((q / -6.478_785_425_830_868e31 - 1.0).abs() < 1e-12, "{q:e}");
    // The quantile is monotone through the median and round-trips through the survival
    // function at probabilities that differ from 1/2 by only a few units in the last place.
    let mut last = f64::NEG_INFINITY;
    for k in 0..40 {
        let p = 0.5 + f64::from(k) * 1.0e-9;
        let t = student_t_ppf(p, 20.0);
        assert!(t >= last, "non-monotone at p={p}");
        last = t;
        if k > 0 {
            assert!(((0.5 - student_t_sf(t, 20.0)) / (p - 0.5) - 1.0).abs() < 1e-6, "p={p}");
        }
    }
}
