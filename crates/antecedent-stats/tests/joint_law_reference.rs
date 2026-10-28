//! Joint-law arithmetic against exact enumeration and rational definitions.
//!
//! `oracle_data/joint_law_reference.py` (frozen in the JSON beside it) does not use the
//! comonotone / countermonotone / Makarov forms of the product.  It enumerates every integer
//! contingency table with the marginal counts, which contains every vertex of the transportation
//! polytope, so the extremes of each coupling-linear functional are exact.  It also evaluates the
//! linear-combination, mixture, correlation-range, spread, exceedance-band, normal-interval and
//! mixture-quantile quantities from their definitions.

use antecedent_core::CancellationToken;
use antecedent_stats::{
    DecisionVerdict, Marginal, MomentSummary, ValueBounds, ValueFunctional,
    composed_exceedance_decision, correlation_range_envelope, covariance_envelope,
    difference_cdf_envelope, draw_spread, functional_envelope, linear_combination_moments,
    linear_variance_envelope, marginal_interval_candidate, mixture_exceedance, mixture_marginal,
    mixture_moments, normal_interval_candidate, product_moment_envelope, sum_cdf_envelope,
};
use serde_json::Value;

fn reference() -> Value {
    serde_json::from_str(include_str!("oracle_data/joint_law_reference.json"))
        .expect("frozen reference parses")
}

fn f(v: &Value) -> f64 {
    v.as_f64().expect("number")
}

fn fs(v: &Value) -> Vec<f64> {
    v.as_array().expect("array").iter().map(f).collect()
}

fn near(got: f64, want: f64, ctx: &str) {
    assert!((got - want).abs() <= 1e-12 * (1.0 + want.abs()), "{ctx}: {got} vs {want}");
}

fn bounds(got: ValueBounds, want: &Value, ctx: &str) {
    let w = fs(want);
    near(got.lower, w[0], &format!("{ctx} lower"));
    near(got.upper, w[1], &format!("{ctx} upper"));
}

fn marginals(case: &Value) -> (Marginal, Marginal) {
    let n = f(&case["n"]);
    let build = |vk: &str, kk: &str| {
        let values = fs(&case[vk]);
        let masses: Vec<f64> = fs(&case[kk]).iter().map(|k| k / n).collect();
        Marginal::from_weighted(&values, &masses).expect("marginal")
    };
    (build("x", "kx"), build("y", "ky"))
}

#[test]
fn coupling_extremes_match_exhaustive_table_enumeration() {
    let token = CancellationToken::new();
    let functionals = [
        ("product", ValueFunctional::Product),
        ("ratio", ValueFunctional::Ratio),
        ("minimum", ValueFunctional::Minimum),
        ("maximum", ValueFunctional::Maximum),
        ("squared_difference", ValueFunctional::SquaredDifference),
    ];
    let (mut accepted, mut refused) = (0, 0);
    for case in reference()["coupling"].as_array().expect("cases") {
        let name = case["name"].as_str().expect("name");
        let (x, y) = marginals(case);
        for (key, functional) in functionals {
            let got = functional_envelope(&x, &y, functional, &token);
            let want = &case["functional"][key];
            if want.as_str() == Some("refuse") {
                assert!(got.is_err(), "{name}/{key}: lattice sign fails, must refuse");
                refused += 1;
            } else {
                bounds(
                    got.unwrap_or_else(|e| panic!("{name}/{key}: {e}")),
                    want,
                    &format!("{name}/{key}"),
                );
                accepted += 1;
            }
        }
        bounds(
            product_moment_envelope(&x, &y, &token).expect("product"),
            &case["product_moment"],
            name,
        );
        bounds(covariance_envelope(&x, &y, &token).expect("cov"), &case["covariance"], name);
        let (a, b) = (f(&case["a"]), f(&case["b"]));
        bounds(
            linear_variance_envelope(a, b, &x, &y, &token).expect("var"),
            &case["linear_variance"],
            name,
        );
        near(x.mean(), f(&case["mean_x"]), name);
        near(y.variance(), f(&case["var_y"]), name);
        for (kind, rows) in [("sum", &case["sum_cdf"]), ("difference", &case["difference_cdf"])] {
            for row in rows.as_array().expect("rows") {
                let r = fs(row);
                let got = if kind == "sum" {
                    sum_cdf_envelope(&x, &y, r[0])
                } else {
                    difference_cdf_envelope(&x, &y, r[0])
                }
                .expect("cdf envelope");
                near(got.lower, r[1], &format!("{name} {kind} t={} lower", r[0]));
                near(got.upper, r[2], &format!("{name} {kind} t={} upper", r[0]));
            }
        }
    }
    assert!(accepted > 60 && refused >= 10, "both regimes are exercised: {accepted}/{refused}");
}

#[test]
fn linear_mixture_and_correlation_moments_match_definitions() {
    let r = reference();
    for case in r["linear"].as_array().expect("linear") {
        let got =
            linear_combination_moments(&fs(&case["coef"]), &fs(&case["mean"]), &fs(&case["cov"]))
                .expect("moments");
        near(got.mean, f(&case["want_mean"]), "linear mean");
        near(got.variance, f(&case["want_variance"]), "linear variance");
    }
    for case in r["mixtures"].as_array().expect("mixtures") {
        let got =
            mixture_moments(&fs(&case["weights"]), &fs(&case["means"]), &fs(&case["variances"]))
                .expect("mixture");
        near(got.mean, f(&case["mean"]), "mixture mean");
        near(got.within, f(&case["within"]), "within");
        near(got.between, f(&case["between"]), "between");
        near(got.variance, f(&case["variance"]), "total variance");
    }
    for case in r["correlation"].as_array().expect("correlation") {
        let got = correlation_range_envelope(
            MomentSummary { mean: f(&case["x_mean"]), standard_deviation: f(&case["x_sd"]) },
            MomentSummary { mean: f(&case["y_mean"]), standard_deviation: f(&case["y_sd"]) },
            f(&case["a"]),
            f(&case["b"]),
            f(&case["lo"]),
            f(&case["hi"]),
        )
        .expect("range");
        bounds(got.product_moment, &case["product_moment"], "range product");
        bounds(got.covariance, &case["covariance"], "range covariance");
        bounds(got.linear_variance, &case["linear_variance"], "range variance");
    }
}

#[test]
fn draw_spread_exceedance_band_and_normal_interval_match_definitions() {
    let r = reference();
    for case in r["spreads"].as_array().expect("spreads") {
        let eff = case["effective_size"].as_f64();
        let got = draw_spread(&fs(&case["draws"]), eff).expect("spread");
        near(got.mean, f(&case["mean"]), "spread mean");
        near(got.variance, f(&case["variance"]), "spread variance");
        near(got.standard_deviation, f(&case["sd"]), "spread sd");
        near(got.mean_standard_error, f(&case["mean_se"]), "spread se");
    }
    for case in r["exceedance"].as_array().expect("exceedance") {
        let got = composed_exceedance_decision(
            &fs(&case["draws"]),
            f(&case["threshold"]),
            f(&case["cutoff"]),
            f(&case["level"]),
            case["effective_size"].as_f64(),
        )
        .expect("decision");
        near(got.probability, f(&case["probability"]), "exceedance probability");
        // The quantile routine is a rational approximation; the band is checked to 1e-9.
        assert!(
            (got.standard_error - f(&case["se"])).abs() < 1e-9,
            "se {} vs {}",
            got.standard_error,
            case["se"]
        );
        assert!((got.band - f(&case["band"])).abs() < 1e-9, "band");
        let verdict = match case["verdict"].as_str().expect("verdict") {
            "act" => DecisionVerdict::Act,
            "withhold" => DecisionVerdict::Withhold,
            _ => DecisionVerdict::Unresolved,
        };
        assert_eq!(got.verdict, verdict, "verdict for {}", case["threshold"]);
    }
    for case in r["normal_interval"].as_array().expect("normal") {
        let got =
            normal_interval_candidate(f(&case["mean"]), f(&case["variance"]), f(&case["level"]))
                .expect("interval");
        let sd = f(&case["variance"]).sqrt();
        let tol = 1e-9 * (1.0 + sd);
        assert!(
            (got.lower - f(&case["lower"])).abs() < tol,
            "lower {} vs {}",
            got.lower,
            case["lower"]
        );
        assert!(
            (got.upper - f(&case["upper"])).abs() < tol,
            "upper {} vs {}",
            got.upper,
            case["upper"]
        );
    }
}

#[test]
fn mixture_marginal_quantile_interval_and_exceedance_match_enumeration() {
    for case in reference()["mixture_marginal"].as_array().expect("mixtures") {
        let weights = fs(&case["weights"]);
        let comps: Vec<Vec<f64>> =
            case["components"].as_array().expect("c").iter().map(fs).collect();
        let refs: Vec<&[f64]> = comps.iter().map(Vec::as_slice).collect();
        let marginal = mixture_marginal(&weights, &refs).expect("mixture marginal");
        let atoms = case["atoms"].as_array().expect("atoms");
        assert_eq!(marginal.atoms().len(), atoms.len());
        for (got, want) in marginal.atoms().iter().zip(atoms) {
            let w = fs(want);
            assert_eq!(got.0, w[0]);
            near(got.1, w[1], "atom mass");
        }
        near(marginal.mean(), f(&case["mean"]), "mixture marginal mean");
        near(marginal.variance(), f(&case["variance"]), "mixture marginal variance");
        let interval = marginal_interval_candidate(&marginal, f(&case["level"])).expect("interval");
        assert_eq!((interval.lower, interval.upper), (f(&case["lower"]), f(&case["upper"])));
        let (p, se) =
            mixture_exceedance(&weights, &refs, f(&case["threshold"])).expect("exceedance");
        near(p, f(&case["exceedance"]), "mixture exceedance");
        near(se, f(&case["exceedance_se"]), "mixture exceedance se");
    }
}
