//! The mapped posterior transfer against an exact rational reference.
//!
//! `oracle_data/mapped_posterior_transfer_reference.py` (frozen in the JSON beside it) takes N
//! equally weighted joint draws of three coefficients and a constant residual variance, forms the
//! mean and the covariance of the finite distribution of the draws (denominator N) and the sample
//! covariance (denominator N - 1) in `fractions.Fraction` arithmetic, and writes, for a complete
//! stable-ID bijection onto a permuted target order, the prior mean, the prior variance
//! `Sigma_ii / sigma2` and the permuted correlation. The product reads the same draws through a
//! distribution artifact, checks the identities and residual variance, and hydrates the prior, so
//! this checks the arithmetic of the route under a three-cycle, coordinates offset by 1e6 with a
//! spread of 1e-3, a correlation within 3e-9 of one, a known variance of 1e4 and the identity
//! order.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::inference::{MappedTransferSpec, mapped_posterior_transfer_with};
use antecedent_contract::{QuantityRole, ScientificQuantity};
use antecedent_estimate::coefficient_covariance_from_draws;
use antecedent_io::distribution_artifact::{
    DistributionArtifact, DistributionCalibration, DistributionIdentity, DistributionMetadata,
    DistributionProvenance, DistributionTrust, DrawAlignment,
};
use antecedent_io::quantity_wire::{DistributionMeaningWire, ScientificQuantityWire};
use antecedent_prob::{GaussianCoefficientPrior, PosteriorQuantityKind, PriorSet, PriorSpec};
use serde_json::Value;

fn quantity(id: &str, functional: &str) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: id.into(),
        variable_name: id.into(),
        role: QuantityRole::Covariate,
        units: "dimensionless".into(),
        population_id: "population-1".into(),
        regime_id: "observed".into(),
        horizon: 0,
        functional_id: functional.into(),
        conditioning: vec![],
        transform_id: "identity".into(),
    }
}

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn close(actual: f64, expected: f64, tol: f64, what: &str) {
    assert!(
        (actual - expected).abs() <= tol * expected.abs().max(1e-12),
        "{what}: {actual} vs {expected}"
    );
}

#[test]
fn mapped_prior_matches_the_exact_moments_under_a_permuted_bijection() {
    let reference: Value =
        serde_json::from_str(include_str!("oracle_data/mapped_posterior_transfer_reference.json"))
            .unwrap();
    for case in reference["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let p = usize::try_from(case["p"].as_u64().unwrap()).unwrap();
        let n = usize::try_from(case["n"].as_u64().unwrap()).unwrap();
        let sigma2 = case["sigma2"].as_f64().unwrap();
        let order: Vec<usize> = case["target_order"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| usize::try_from(v.as_u64().unwrap()).unwrap())
            .collect();
        let rows: Vec<Vec<f64>> = case["draws"].as_array().unwrap().iter().map(floats).collect();

        let source_quantities: Vec<ScientificQuantity> = (0..p)
            .map(|i| quantity(&format!("source:c{i}"), "coefficient"))
            .chain([quantity("source:sigma2", "residual_variance")])
            .collect();
        let identity = DistributionIdentity::new(
            DistributionMeaningWire::ParameterPosterior,
            &source_quantities,
            DrawAlignment::Joint,
            DistributionProvenance {
                source_id: "source".into(),
                provider_id: "finite-draws".into(),
                rng_id: "frozen".into(),
                snapshot_id: "source-snapshot".into(),
                causal_contract_id: "contract".into(),
            },
        )
        .unwrap();
        let artifact = DistributionArtifact::new(
            DistributionMetadata {
                version: 1,
                identity: identity.clone(),
                axes: ["draw".into(), "quantity".into()],
                shape: [n, p + 1],
                weights: None,
                supported: None,
                calibration: DistributionCalibration::PointOnly,
                trust: DistributionTrust::Unverified,
                legacy_posterior: None,
                legacy_bindings: None,
            },
            rows.iter().flatten().copied().collect(),
        )
        .unwrap();
        let bytes = artifact.to_bytes("source-posterior").unwrap();

        // Target coefficient k is source coefficient order[k].
        let targets: Vec<ScientificQuantityWire> = (0..p)
            .map(|k| {
                ScientificQuantityWire::from(&quantity(&format!("target:c{k}"), "coefficient"))
            })
            .collect();
        let pairs: Vec<(String, String)> =
            (0..p).map(|k| (format!("source:c{}", order[k]), format!("target:c{k}"))).collect();
        let mut baseline = PriorSet::new();
        baseline.push(PriorSpec::GaussianCoefficients(GaussianCoefficientPrior::isotropic(p, 1.0)));
        baseline.push(PriorSpec::KnownResidualVariance(sigma2));
        let prior = mapped_posterior_transfer_with(
            &bytes,
            &MappedTransferSpec {
                expected_source: &identity,
                source_to_target: &pairs,
                target_quantities: &targets,
                target_snapshot_id: "target-snapshot",
                target_contract_id: "contract",
                baseline: &baseline,
            },
        )
        .unwrap();

        let coef = prior.gaussian_coefficients().unwrap();
        let (want_mean, want_var) = (floats(&case["prior_mean"]), floats(&case["prior_variance"]));
        for k in 0..p {
            close(coef.mean[k], want_mean[k], 1e-9, &format!("{name} mean {k}"));
            close(coef.variance[k], want_var[k], 1e-6, &format!("{name} variance {k}"));
        }
        let correlation = prior.coefficient_correlation().expect("dense V0 carried");
        for (k, row) in case["correlation"].as_array().unwrap().iter().enumerate() {
            for (l, want) in floats(row).iter().enumerate() {
                // 1 - rho is 3e-9 in the collinear case, so compare absolutely.
                assert!(
                    (correlation.matrix()[k * p + l] - want).abs() <= 1e-9,
                    "{name} correlation ({k},{l}): {} vs {want}",
                    correlation.matrix()[k * p + l]
                );
            }
        }

        // The posterior-artifact covariance of the same draws (denominator N - 1), column-major.
        let quantities: Vec<PosteriorQuantityKind> =
            (0..p).map(|index| PosteriorQuantityKind::Coefficient { index, name: None }).collect();
        let column_major: Vec<f64> = (0..p).flat_map(|i| rows.iter().map(move |r| r[i])).collect();
        let sample = coefficient_covariance_from_draws(&quantities, &column_major, n)
            .unwrap()
            .expect("more draws than coefficients");
        for (i, row) in case["sample_covariance"].as_array().unwrap().iter().enumerate() {
            for (j, want) in floats(row).iter().enumerate() {
                close(sample[i * p + j], *want, 1e-7, &format!("{name} sample cov ({i},{j})"));
            }
        }
    }
}
