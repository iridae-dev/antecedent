//! The prior-conflict alpha policy, its composition and the Gaussian divergence against
//! 80-digit evaluation of their definitions (`oracle_data/conflict_policy_reference.py`).
//!
//! Covers every combination of alpha in {0, 0.3, 1}, p-value (absent, 0, exactly at the
//! threshold, one ulp-scale above it, 1), divergence (absent, 0, 1e-12 .. 800) and four
//! threshold/scale settings; end-to-end shrink-then-compose for one-parameter Gaussian priors;
//! and the divergence for variance ratios from 1e-9 to 1e9 and relative differences near 1e-16.

use std::sync::Arc;

use antecedent_prob::{
    ExternalPriorSource, ExternalPriorWeight, GaussianCoefficientPrior, PriorSet, PriorSpec,
};
use antecedent_stats::gaussian_kl;
use antecedent_validate::conflict::{ConflictPolicy, ConflictSignals, apply_conflict_and_compose};
use serde_json::Value;

fn reference() -> Value {
    serde_json::from_str(include_str!("oracle_data/conflict_policy_reference.json"))
        .expect("frozen reference")
}

fn opt(value: &Value) -> Option<f64> {
    value.as_f64()
}

fn gauss(mean: f64, variance: f64) -> PriorSet {
    let mut set = PriorSet::new();
    set.push(PriorSpec::GaussianCoefficients(
        GaussianCoefficientPrior::shared(1, mean, variance).expect("prior"),
    ));
    set
}

#[test]
fn alpha_shrink_equals_the_stated_formula() {
    let reference = reference();
    for case in reference["shrink"].as_array().expect("cases") {
        let policy = ConflictPolicy::try_new(
            case["p_min"].as_f64().expect("p_min"),
            case["kl_scale"].as_f64().expect("kl_scale"),
        )
        .expect("policy");
        let alpha = case["alpha"].as_f64().expect("alpha");
        let got = policy.shrink_alpha(alpha, opt(&case["p"]), opt(&case["kl"]));
        let want = case["applied"].as_f64().expect("applied");
        assert!(
            (got - want).abs() <= 1e-15 * alpha.max(1.0) + 1e-14 * want.abs(),
            "{case}: {got} vs {want}"
        );
        assert!(got <= alpha, "never increases alpha: {case}");
    }
}

#[test]
fn invalid_signals_drop_the_source_instead_of_passing_it() {
    let policy = ConflictPolicy::new();
    for (p, kl) in [
        (Some(f64::NAN), None),
        (Some(f64::INFINITY), Some(0.0)),
        (Some(1.5), None),
        (Some(-0.1), None),
        (None, Some(f64::NAN)),
        (None, Some(f64::INFINITY)),
        (Some(0.9), Some(-1.0)),
    ] {
        assert_eq!(policy.shrink_alpha(0.7, p, kl), 0.0, "p {p:?} kl {kl:?}");
    }
    // Absent signals stay a no-op.
    assert!((policy.shrink_alpha(0.7, None, None) - 0.7).abs() < 1e-15);
}

#[test]
fn shrink_then_compose_equals_the_precision_weighted_reference() {
    let reference = reference();
    for case in reference["compose"].as_array().expect("cases") {
        let name = case["name"].as_str().expect("name");
        let base = case["baseline"].as_array().expect("baseline");
        let baseline = gauss(base[0].as_f64().expect("m"), base[1].as_f64().expect("v"));
        let mut sources = Vec::new();
        let mut signals = Vec::new();
        for (i, src) in case["sources"].as_array().expect("sources").iter().enumerate() {
            sources.push(ExternalPriorSource {
                id: Arc::from(format!("s{i}")),
                prior: gauss(
                    src["mean"].as_f64().expect("mean"),
                    src["variance"].as_f64().expect("variance"),
                ),
                weight: ExternalPriorWeight::power(src["alpha"].as_f64().expect("alpha"))
                    .expect("weight"),
                ess: None,
            });
            signals.push(ConflictSignals { p_value: opt(&src["p"]), kl: opt(&src["kl"]) });
        }
        let (composed, summary) =
            apply_conflict_and_compose(&sources, &baseline, &ConflictPolicy::new(), &signals)
                .expect("compose");
        let applied = case["applied"].as_array().expect("applied");
        for (i, want) in applied.iter().enumerate() {
            let want = want.as_f64().expect("alpha");
            assert!((summary.alphas_applied[i] - want).abs() <= 1e-14, "{name} alpha {i}");
            assert!((composed.alphas_applied[i] - want).abs() <= 1e-14, "{name} composed {i}");
        }
        let coefficients = composed.prior.gaussian_coefficients().expect("coefficients");
        let (mean, variance) = (coefficients.mean[0], coefficients.variance[0]);
        let (want_mean, want_variance) =
            (case["mean"].as_f64().expect("mean"), case["variance"].as_f64().expect("variance"));
        assert!((mean - want_mean).abs() <= 1e-12 * want_mean.abs().max(1.0), "{name}: {mean}");
        assert!((variance - want_variance).abs() <= 1e-12 * want_variance, "{name}: {variance}");
    }
}

#[test]
fn gaussian_divergence_matches_high_precision_across_variance_ratios() {
    let reference = reference();
    for case in reference["kl"].as_array().expect("cases") {
        let f = |key: &str| case[key].as_f64().expect("number");
        let got = gaussian_kl(f("m0"), f("v0"), f("m1"), f("v1")).expect("kl");
        let want = f("kl");
        assert!(
            (got - want).abs() <= 1e-12 * want.abs() + 1e-300,
            "{case}: {got:e} vs {want:e}"
        );
    }
}
