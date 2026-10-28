//! Joint Bayesian source-target transport against an exact rational reference.
//!
//! `oracle_data/joint_bayesian_transport_reference.py` (frozen in the JSON beside it) solves the conjugate
//! Gaussian posterior of the model by exact Gauss-Jordan elimination in Fractions on dyadic inputs:
//! precision P0 + sum Z'Z / sigma^2, mean precision^-1 (P0 m0 + sum Z'y / sigma^2), and the exact
//! joint mean and covariance of every source effect and the target effect. Designs cover a dense
//! prior with a nonzero mean, a covariate sitting far from zero, a 2^20 noise-variance ratio between
//! unbalanced sources, a nearly flat prior, collinear covariates under a proper prior and the
//! smallest balanced sample. This test builds the same declaration for the product and requires its
//! reported posterior and effect moments to match, and the iid draws to reproduce the effect
//! moments within Monte Carlo error.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(clippy::cast_precision_loss, reason = "small dense fixtures")]

use std::sync::Arc;

use antecedent_core::{ExecutionContext, VariableId};
use antecedent_estimate::joint_bayesian_transport::{
    DETAIL_SINGULAR_FIT, DataIdentity, GaussianPrior, JointPriors, JointTransportModel,
    JointTransportOptions, PriorProvenance, SourceData, SourceDependence, SourceSharing,
    TargetData, TransportGraphClass, VaryingBlock, fit_joint_bayesian_transport,
};
use antecedent_identify::{
    PopulationFactor, TransportCertificate, TransportFormula, TransportIdentification,
};
use serde_json::Value;

const DRAWS: usize = 40_000;

fn num(v: &Value) -> f64 {
    v.as_f64().unwrap()
}

fn vec_of(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(num).collect()
}

fn factor() -> PopulationFactor {
    PopulationFactor {
        population: Arc::from("source"),
        regime: None,
        variables: Arc::from([]),
        conditioned_on: Arc::from([]),
        interventions: Arc::from([]),
    }
}

fn identified(features: &[u32]) -> TransportIdentification {
    TransportIdentification::Transportable {
        formula: TransportFormula::Standardize {
            over: features.iter().map(|&f| VariableId::from_raw(f)).collect::<Vec<_>>().into(),
            source_response: factor(),
            target_law: factor(),
        },
        certificate: TransportCertificate {
            rule: Arc::from("standardization"),
            selection_targets: Arc::from([]),
            premises: Arc::from([]),
        },
    }
}

fn prior(mean: &Value, cov: &Value) -> GaussianPrior {
    GaussianPrior {
        mean: vec_of(mean),
        covariance: cov.as_array().unwrap().iter().flat_map(vec_of).collect(),
        provenance: PriorProvenance::Declared,
    }
}

fn near(got: f64, want: f64, scale: f64, tolerance: f64, what: &str) {
    assert!((got - want).abs() <= tolerance * (want.abs().max(scale)), "{what}: {got} vs {want}");
}

#[test]
fn posterior_and_effect_moments_match_the_exact_reference() {
    let reference: Value =
        serde_json::from_str(include_str!("oracle_data/joint_bayesian_transport_reference.json"))
            .unwrap();
    let designs = reference["designs"].as_array().unwrap();
    assert_eq!(designs.len(), 6);
    for design in designs {
        let name = design["name"].as_str().unwrap();
        let p = design["features"].as_u64().map_or(1, |v| v as usize);
        let features: Vec<u32> = (0..p as u32).map(|j| 7 + j).collect();
        let varying = match design["varying"].as_str().unwrap() {
            "intercept" => VaryingBlock::Intercept,
            _ => VaryingBlock::InterceptAndCovariates,
        };
        let sharing = if design["shared"].as_bool().unwrap() {
            SourceSharing::SharedVaryingBlock
        } else {
            SourceSharing::IndependentVaryingBlocks
        };
        let sources: Vec<SourceData> = design["sources"]
            .as_array()
            .unwrap()
            .iter()
            .enumerate()
            .map(|(s, src)| SourceData {
                id: format!("s{s}"),
                identity: DataIdentity {
                    snapshot_digest: format!("snap-{s}"),
                    datum_ids: (0..src["y"].as_array().unwrap().len())
                        .map(|i| format!("s{s}-{i}"))
                        .collect(),
                },
                treatment: src["a"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_bool().unwrap())
                    .collect(),
                outcome: vec_of(&src["y"]),
                covariates: src["x"].as_array().unwrap().iter().map(vec_of).collect(),
                noise_variance: num(&src["noise"]),
            })
            .collect();
        let target_cols: Vec<Vec<f64>> =
            design["target"]["x"].as_array().unwrap().iter().map(vec_of).collect();
        let target = TargetData {
            identity: DataIdentity {
                snapshot_digest: "target".into(),
                datum_ids: (0..target_cols[0].len()).map(|j| format!("t{j}")).collect(),
            },
            rows: target_cols[0].len(),
            covariates: target_cols,
        };
        let model = JointTransportModel {
            graph: TransportGraphClass::FixedDag,
            features: features.clone(),
            varying,
            sharing,
            dependence: SourceDependence::IndependentSamples,
            priors: JointPriors {
                invariant: prior(&design["inv_mean"], &design["inv_cov"]),
                varying: prior(&design["var_mean"], &design["var_cov"]),
            },
            max_unsupported_mass: 0.0,
            conflict_z_threshold: 3.0,
        };
        let result = fit_joint_bayesian_transport(
            &identified(&features),
            &model,
            &sources,
            Some(&target),
            &JointTransportOptions { draws: DRAWS, seed: 20_261_028 },
            &ExecutionContext::for_tests(1),
        );
        if design["refused_singular"].as_bool().unwrap_or(false) {
            let refusal = result.expect_err("a design the data cannot separate must be refused");
            assert_eq!(refusal.detail, DETAIL_SINGULAR_FIT, "{name}: {refusal:?}");
            continue;
        }
        let fit = result.unwrap_or_else(|e| panic!("{name}: {e:?}"));
        let d = design["dim"].as_u64().unwrap() as usize;
        assert_eq!(fit.posterior_mean.len(), d, "{name}");
        let scale_of = |i: usize| num(&design["posterior_covariance"][i][i]["value"]).sqrt();
        for i in 0..d {
            near(
                fit.posterior_mean[i],
                num(&design["posterior_mean"][i]["value"]),
                scale_of(i),
                1e-8,
                &format!("{name} mean {i}"),
            );
            for j in 0..d {
                let want = num(&design["posterior_covariance"][i][j]["value"]);
                near(
                    fit.posterior_covariance[i * d + j],
                    want,
                    scale_of(i) * scale_of(j),
                    1e-8,
                    &format!("{name} cov {i},{j}"),
                );
            }
        }
        let e = design["effect_mean"].as_array().unwrap().len();
        assert_eq!(fit.effect_means.len(), e, "{name}");
        let escale = |i: usize| num(&design["effect_covariance"][i][i]["value"]).sqrt();
        for i in 0..e {
            near(
                fit.effect_means[i],
                num(&design["effect_mean"][i]["value"]),
                escale(i),
                1e-8,
                &format!("{name} effect {i}"),
            );
            for j in 0..e {
                near(
                    fit.effect_covariance[i * e + j],
                    num(&design["effect_covariance"][i][j]["value"]),
                    escale(i) * escale(j),
                    1e-8,
                    &format!("{name} effect cov {i},{j}"),
                );
            }
        }
        near(
            fit.target_effect_mean,
            num(&design["effect_mean"][e - 1]["value"]),
            escale(e - 1),
            1e-8,
            name,
        );
        near(
            fit.target_effect_variance,
            num(&design["effect_covariance"][e - 1][e - 1]["value"]),
            0.0,
            1e-8,
            name,
        );
        // The iid draws reproduce the exact target-effect moments within Monte Carlo error.
        let column = fit.draws.column("effect.target").expect("target effect column");
        let draws = fit.draws.coordinate(column);
        let n = draws.len() as f64;
        let mean = draws.iter().sum::<f64>() / n;
        let var = draws.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n;
        let sd = num(&design["effect_covariance"][e - 1][e - 1]["value"]).sqrt();
        assert!((mean - fit.target_effect_mean).abs() < 5.0 * sd / n.sqrt(), "{name}: draw mean");
        assert!(
            (var / (sd * sd) - 1.0).abs() < 0.06,
            "{name}: draw variance ratio {}",
            var / (sd * sd)
        );
    }
}
