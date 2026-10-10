//! Precision of actual portable posterior quantiles, not frequentist coverage.
//! IID aligned draws Z~N(0,1), W=.6Z+.8E~N(0,1), Q=2+3W.
//! Actual export and independently bound consume precede original type-7 summaries.
//! Exact normal quantiles and their asymptotic MC variance are independent oracles.
#![allow(clippy::cast_precision_loss, reason = "bounded simulation dimensions")]
#[path = "../../antecedent/tests/common/calibration.rs"]
mod calibration;
#[path = "../../antecedent-estimate/tests/common/candidate_calibration.rs"]
mod candidate;
use antecedent_core::{QuantityRole, ScientificQuantity};
use antecedent_io::distribution_artifact::{
    DistributionArtifact, DistributionCalibration, DistributionIdentity, DistributionMetadata,
    DistributionProvenance, DistributionTrust, DrawAlignment,
};
use antecedent_io::quantity_wire::DistributionMeaningWire;
use calibration::{grid_n, grid_seed, map_replicates, n_sim};

fn identity(meaning: DistributionMeaningWire) -> DistributionIdentity {
    let quantities = ["z", "q"].map(|name| ScientificQuantity {
        variable_id: format!("schema:{name}"),
        variable_name: name.into(),
        role: QuantityRole::Outcome,
        units: "declared_parameter_units".into(),
        population_id: "conditional_model".into(),
        regime_id: "posterior".into(),
        horizon: 0,
        functional_id: "parameter".into(),
        conditioning: vec![],
        transform_id: "identity".into(),
    });
    DistributionIdentity::new(
        meaning,
        &quantities,
        DrawAlignment::Joint,
        DistributionProvenance {
            source_id: "independent_gaussian_oracle".into(),
            provider_id: "test_draws".into(),
            rng_id: "independent_box_muller".into(),
            snapshot_id: "known_conditional_law".into(),
            causal_contract_id: "declared_model_only".into(),
        },
    )
    .unwrap()
}

#[test]
#[ignore = "calibration: final measurement only"]
fn posterior_type7_quantile_precision() {
    let draws = grid_n(4096);
    for meaning in [
        DistributionMeaningWire::ParameterPosterior,
        DistributionMeaningWire::CausalFunctionalPosterior,
    ] {
        for (mass, z) in [(0.9_f64, 1.644_853_626_951_472_2_f64), (0.95, 1.959_963_984_540_054)] {
            let tail = (1. - mass) / 2.;
            let density = (-z * z / 2.).exp() / std::f64::consts::TAU.sqrt();
            let mc_scale = (tail * (1. - tail) / draws as f64).sqrt() / density;
            let results = map_replicates(n_sim(), |rep| {
                let mut rng = candidate::Generator::new(grid_seed(0x6abc_0000 + rep));
                let mut values = Vec::with_capacity(2 * draws);
                for _ in 0..draws {
                    let a = rng.normal();
                    values.extend([a, 2. + 3. * (0.6 * a + 0.8 * rng.normal())]);
                }
                let expected = identity(meaning);
                let original = DistributionArtifact::new(
                    DistributionMetadata {
                        version: 1,
                        identity: expected.clone(),
                        axes: ["draw".into(), "quantity".into()],
                        shape: [draws, 2],
                        weights: None,
                        supported: None,
                        calibration: DistributionCalibration::Unmeasured,
                        trust: DistributionTrust::Unverified,
                        legacy_posterior: None,
                        legacy_bindings: None,
                    },
                    values,
                )
                .unwrap();
                let bytes = original.to_bytes("independent_gaussian_posterior").unwrap();
                let consumed = DistributionArtifact::from_bytes(&bytes, &expected).unwrap();
                let (a, b) = consumed.posterior_equal_tailed_interval(0, mass).unwrap();
                let (c, d) = consumed.posterior_equal_tailed_interval(1, mass).unwrap();
                [
                    (a + z) / mc_scale,
                    (b - z) / mc_scale,
                    ((c - 2.) / 3. + z) / mc_scale,
                    ((d - 2.) / 3. - z) / mc_scale,
                ]
            });
            let mut bias = [0.; 4];
            let mut squared = [0.; 4];
            for result in &results {
                for i in 0..4 {
                    bias[i] += result[i] / results.len() as f64;
                    squared[i] += result[i] * result[i] / results.len() as f64;
                }
            }
            for i in 0..4 {
                assert!(bias[i].abs() < 0.3, "posterior quantile MC bias {bias:?}");
                assert!(squared[i].sqrt() < 1.4, "posterior quantile MC RMSE {squared:?}");
            }
            println!(
                "diagnostic-precision posterior_type7_quantile_precision meaning={meaning:?} mass={mass} draws={draws} repetitions={} standardized_bias={bias:?} standardized_squared_error={squared:?}; posterior_mass_only_no_sampling_coverage",
                results.len()
            );
        }
    }
}
