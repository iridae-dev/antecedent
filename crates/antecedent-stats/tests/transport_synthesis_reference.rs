//! Transport-aware synthesis against exact rational and 50-digit references.
//!
//! `oracle_data/transport_synthesis_reference.py` (frozen in the JSON beside it) solves the
//! transport-aware covariance (sampling covariance plus the shared transport variance on the
//! diagonal and between summaries that share a transport key) exactly in Fractions, evaluates the
//! hierarchical posterior over a discrete spread prior from the Gaussian marginal-likelihood
//! definition with explicit determinants and inverses at 50 digits, and the transported-prior
//! precision-weighted update exactly. The test builds the same inputs for the product and requires
//! agreement, including the heterogeneity, the variance without transport, the dropped-spread
//! posterior of a one-point prior, and the documented refusal of a correlation beyond the
//! singularity threshold.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_stats::synthesis::{MeanPrior, SynthesisError, TauPrior};
use antecedent_stats::synthesis_transport::{
    SummaryOrigin, TransportAwareSummary, TransportedSynthesisError, borrow_transported,
    pool_transported, transported_prior_update,
};
use serde_json::Value;

fn reference() -> Value {
    serde_json::from_str(include_str!("oracle_data/transport_synthesis_reference.json")).unwrap()
}

fn num(v: &Value) -> f64 {
    v.as_f64().unwrap()
}

fn near(got: f64, want: f64, rel: f64, abs: f64, what: &str) {
    assert!((got - want).abs() <= rel * want.abs() + abs, "{what}: {got} vs {want}");
}

fn summary(s: &Value) -> TransportAwareSummary {
    let origin = match s["key"].as_str() {
        None => SummaryOrigin::Native,
        Some(key) => SummaryOrigin::Transported {
            transport_key: key.to_owned(),
            transport_variance: num(&s["tau2"]),
        },
    };
    TransportAwareSummary { estimate: num(&s["y"]), sampling_variance: num(&s["var"]), origin }
}

#[test]
fn pooling_matches_exact_generalized_least_squares() {
    let reference = reference();
    let pools = reference["pools"].as_array().unwrap();
    assert_eq!(pools.len(), 6);
    for case in pools {
        let name = case["name"].as_str().unwrap();
        let summaries: Vec<TransportAwareSummary> =
            case["summaries"].as_array().unwrap().iter().map(summary).collect();
        let base: Option<Vec<f64>> = case["base"]
            .as_array()
            .map(|rows| rows.iter().flat_map(|r| r.as_array().unwrap().iter().map(num)).collect());
        let got = pool_transported(&summaries, base.as_deref());
        if case["refused_as_singular"].as_bool().unwrap() {
            assert!(
                matches!(
                    got,
                    Err(TransportedSynthesisError::Synthesis(SynthesisError::Singular { .. }))
                ),
                "{name}: {got:?}"
            );
            continue;
        }
        let got = got.unwrap();
        let k = summaries.len();
        near(got.pooled.estimate, num(&case["estimate"]["value"]), 1e-10, 1e-12, name);
        near(got.pooled.variance, num(&case["variance"]["value"]), 1e-10, 0.0, name);
        near(got.pooled.heterogeneity_q, num(&case["heterogeneity"]["value"]), 1e-8, 1e-12, name);
        near(
            got.variance_without_transport,
            num(&case["variance_without_transport"]["value"]),
            1e-10,
            0.0,
            name,
        );
        near(
            got.transport_variance_added,
            num(&case["transport_variance_added"]["value"]),
            1e-8,
            1e-12 * got.pooled.variance,
            name,
        );
        assert_eq!(got.pooled.degrees_of_freedom, k - 1, "{name}");
        for i in 0..k {
            near(got.pooled.weights[i], num(&case["weights"][i]["value"]), 1e-9, 1e-12, name);
            for j in 0..k {
                near(
                    got.covariance[i * k + j],
                    num(&case["covariance"][i][j]["value"]),
                    1e-14,
                    0.0,
                    name,
                );
            }
        }
    }
}

#[test]
fn borrowing_matches_the_marginal_likelihood_definition_at_fifty_digits() {
    let reference = reference();
    for case in reference["borrows"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let eff: Vec<f64> =
            case["effective_variances"].as_array().unwrap().iter().map(num).collect();
        let summaries: Vec<TransportAwareSummary> = case["estimates"]
            .as_array()
            .unwrap()
            .iter()
            .zip(&eff)
            .zip(case["transport_variances"].as_array().unwrap())
            .enumerate()
            .map(|(i, ((y, &e), tau2))| match tau2.as_f64() {
                None => TransportAwareSummary {
                    estimate: num(y),
                    sampling_variance: e,
                    origin: SummaryOrigin::Native,
                },
                Some(t) => TransportAwareSummary {
                    estimate: num(y),
                    sampling_variance: e - t,
                    origin: SummaryOrigin::Transported {
                        transport_key: format!("k{i}"),
                        transport_variance: t,
                    },
                },
            })
            .collect();
        let prior = match case["prior"].as_array() {
            None => MeanPrior::Flat,
            Some(p) => MeanPrior::Normal { mean: num(&p[0]), variance: num(&p[1]) },
        };
        let tau = TauPrior {
            support: case["tau_support"].as_array().unwrap().iter().map(num).collect(),
            probabilities: case["tau_probabilities"].as_array().unwrap().iter().map(num).collect(),
        };
        let got = borrow_transported(&summaries, prior, &tau).unwrap();
        for (g, w) in got.effective_variances.iter().zip(&eff) {
            near(*g, *w, 1e-15, 0.0, name);
        }
        let p = &got.posterior;
        near(p.mean, num(&case["mean"]), 1e-9, 1e-12, name);
        near(p.variance, num(&case["variance"]), 1e-8, 1e-12, name);
        for (i, w) in case["study_means"].as_array().unwrap().iter().enumerate() {
            near(p.study_means[i], num(w), 1e-9, 1e-12, name);
            near(p.study_variances[i], num(&case["study_variances"][i]), 1e-7, 1e-10, name);
        }
        for (i, w) in case["tau_posterior"].as_array().unwrap().iter().enumerate() {
            near(p.tau_posterior[i], num(w), 1e-9, 1e-12, name);
        }
    }
}

#[test]
fn transported_prior_update_matches_exact_precision_weighting() {
    let reference = reference();
    for case in reference["updates"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let prior = summary(&case["prior"]);
        let likelihood: Vec<TransportAwareSummary> =
            case["likelihood"].as_array().unwrap().iter().map(summary).collect();
        let got = transported_prior_update(&prior, &likelihood).unwrap();
        near(got.mean, num(&case["mean"]["value"]), 1e-14, 0.0, name);
        near(got.variance, num(&case["variance"]["value"]), 1e-14, 0.0, name);
        near(got.prior_variance, num(&case["prior_variance"]["value"]), 1e-14, 0.0, name);
        near(got.prior_weight, num(&case["prior_weight"]["value"]), 1e-14, 0.0, name);
        near(
            got.likelihood_precision,
            num(&case["likelihood_precision"]["value"]),
            1e-14,
            0.0,
            name,
        );
    }
}

#[test]
fn declarations_the_route_cannot_honour_are_refused_not_ignored() {
    let native = |y, v| TransportAwareSummary {
        estimate: y,
        sampling_variance: v,
        origin: SummaryOrigin::Native,
    };
    let carried = |y, v, key: &str, t| TransportAwareSummary {
        estimate: y,
        sampling_variance: v,
        origin: SummaryOrigin::Transported { transport_key: key.to_owned(), transport_variance: t },
    };
    let tau = TauPrior { support: vec![0.5, 1.0], probabilities: vec![0.5, 0.5] };
    // Two summaries of one transport cannot be borrowed over as independent studies.
    let shared = [native(1.0, 1.0), carried(2.0, 1.0, "t", 0.5), carried(3.0, 1.0, "t", 0.5)];
    assert!(matches!(
        borrow_transported(&shared, MeanPrior::Flat, &tau),
        Err(TransportedSynthesisError::SharedTransportNeedsCovariance { .. })
    ));
    // One transport declared with two variances is a contradiction, not a default.
    let clash = [carried(2.0, 1.0, "t", 0.5), carried(3.0, 1.0, "t", 0.75)];
    assert!(matches!(
        pool_transported(&clash, None),
        Err(TransportedSynthesisError::InconsistentTransportVariance { .. })
    ));
    // A supplied sampling covariance must agree with the stated variances on its diagonal.
    let diag = [native(1.0, 1.0), native(2.0, 1.0)];
    assert!(matches!(
        pool_transported(&diag, Some(&[1.5, 0.0, 0.0, 1.0])),
        Err(TransportedSynthesisError::SamplingCovarianceDiagonalMismatch { index: 0 })
    ));
    // A native prior and a transported likelihood both refuse.
    assert!(matches!(
        transported_prior_update(&native(1.0, 1.0), &[native(1.0, 1.0)]),
        Err(TransportedSynthesisError::PriorRoleMismatch(_))
    ));
    assert!(matches!(
        transported_prior_update(&carried(1.0, 1.0, "t", 1.0), &[carried(1.0, 1.0, "u", 1.0)]),
        Err(TransportedSynthesisError::PriorRoleMismatch(_))
    ));
    // A negative transport variance is invalid, never read as zero.
    assert!(matches!(
        pool_transported(&[native(1.0, 1.0), carried(1.0, 1.0, "t", -0.1)], None),
        Err(TransportedSynthesisError::InvalidTransportVariance { index: 1 })
    ));
}
