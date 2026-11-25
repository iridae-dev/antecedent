//! Facade for the legacy posterior conversion.
//!
//! [`convert_legacy_posterior`] returns the artifact of
//! [`antecedent_io::distribution_artifact::convert_legacy_posterior`]. Draw meaning is the
//! identity supplied with the call: parameter, predictive, bootstrap and structural
//! (causal-functional) draws stay on that meaning. A posterior that does not name the
//! outcome quantities that meaning requires is refused by the conversion. This module
//! does not hash a collection.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_io::IoError;

/// Source-to-target quantity binding of a legacy posterior conversion.
pub use antecedent_io::distribution_artifact::LegacyPosteriorBinding;

/// Convert a legacy posterior artifact into an aligned joint distribution.
///
/// The result is [`antecedent_io::distribution_artifact::convert_legacy_posterior`] on
/// these arguments: the same draws, receipt, bindings, calibration, trust and refusals.
/// `expected.semantic` is stored as given.
///
/// # Errors
///
/// The errors of [`antecedent_io::distribution_artifact::convert_legacy_posterior`].
/// Unnamed outcome quantities and unnamed coefficients are
/// `distribution_meaning_mismatch`. Predictive, bootstrap and structural meanings that
/// the source quantities do not support are the same refusal. Unresolved structural
/// mass is refused before a meaning is granted.
pub fn convert_legacy_posterior(
    legacy: &crate::io::EncodedArtifact,
    expected: crate::io::distribution_artifact::DistributionIdentity,
    bindings: &[LegacyPosteriorBinding],
) -> Result<crate::analysis::inverse_query::DistributionArtifact, IoError> {
    antecedent_io::distribution_artifact::convert_legacy_posterior(legacy, expected, bindings)
}

#[cfg(test)]
mod tests {
    use super::convert_legacy_posterior;
    use antecedent_contract::{QuantityRole, ScientificQuantity};
    use antecedent_io::EncodedArtifact;
    use antecedent_io::distribution_artifact::convert_legacy_posterior as convert_core;
    use antecedent_io::distribution_artifact::{
        DistributionIdentity, DistributionProvenance, DrawAlignment, LegacyPosteriorBinding,
    };
    use antecedent_io::posterior::{
        CausalPosteriorWire, PosteriorQuantityWire, encode_posterior_artifact,
    };
    use antecedent_io::quantity_wire::{DistributionMeaningWire, ScientificQuantityWire};

    const MEANINGS: [DistributionMeaningWire; 7] = [
        DistributionMeaningWire::ParameterPosterior,
        DistributionMeaningWire::CausalFunctionalPosterior,
        DistributionMeaningWire::PosteriorPredictive,
        DistributionMeaningWire::InterventionalPredictive,
        DistributionMeaningWire::EstimatorSampling,
        DistributionMeaningWire::Bootstrap,
        DistributionMeaningWire::EmpiricalOutcome,
    ];

    fn quantity(name: &str, role: QuantityRole, functional: &str) -> ScientificQuantity {
        ScientificQuantity {
            variable_id: format!("schema:{name}"),
            variable_name: name.into(),
            role,
            units: "dimensionless".into(),
            population_id: "target".into(),
            regime_id: "do(a=1)".into(),
            horizon: 0,
            functional_id: functional.into(),
            conditioning: vec![],
            transform_id: "identity".into(),
        }
    }

    fn identity(
        semantic: DistributionMeaningWire,
        quantities: &[ScientificQuantity],
        source_id: &str,
    ) -> DistributionIdentity {
        DistributionIdentity::new(
            semantic,
            quantities,
            DrawAlignment::Joint,
            DistributionProvenance {
                source_id: source_id.into(),
                provider_id: "legacy-conversion".into(),
                rng_id: "source-posterior-draws".into(),
                snapshot_id: "checked-snapshot".into(),
                causal_contract_id: "checked-contract".into(),
            },
        )
        .unwrap()
    }

    fn paired() -> CausalPosteriorWire {
        CausalPosteriorWire {
            quantities: vec![
                PosteriorQuantityWire::Coefficient { index: 0, name: Some("beta_x".into()) },
                PosteriorQuantityWire::Coefficient { index: 1, name: Some("beta_y".into()) },
            ],
            n_draws: 3,
            mean: vec![1.0, 2.0],
            sd: vec![1.0, 2.0],
            q025: vec![0.05, 0.1],
            q975: vec![1.95, 3.9],
            identification: "NonparametricallyIdentified".into(),
            unidentified_mass: 0.0,
            subsampled_out_mass: 0.0,
            backend_id: "conjugate".into(),
            converged: true,
            hessian_condition: 1.0,
            draws_encoding: "f64_le_colmajor".into(),
            treatment_contrast: None,
            preparation: None,
        }
    }

    fn encode(meta: &CausalPosteriorWire) -> EncodedArtifact {
        encode_posterior_artifact(
            meta,
            &[0.0, 1.0, 2.0, 0.0, 2.0, 4.0],
            "source-posterior",
            "2.2.0",
        )
        .unwrap()
    }

    fn bind(
        sources: &[PosteriorQuantityWire],
        targets: &[ScientificQuantity],
    ) -> Vec<LegacyPosteriorBinding> {
        sources
            .iter()
            .zip(targets)
            .map(|(source, target)| LegacyPosteriorBinding {
                source: source.clone(),
                target: ScientificQuantityWire::from(target),
            })
            .collect()
    }

    fn assert_same(
        encoded: &EncodedArtifact,
        expected: &DistributionIdentity,
        bindings: &[LegacyPosteriorBinding],
    ) {
        let facade = convert_legacy_posterior(encoded, expected.clone(), bindings);
        let core = convert_core(encoded, expected.clone(), bindings);
        assert_eq!(facade, core);
        if let Ok(artifact) = &facade {
            assert_eq!(
                artifact.to_bytes("converted").unwrap(),
                core.unwrap().to_bytes("converted").unwrap()
            );
        }
    }

    #[test]
    fn facade_output_matches_the_io_core_conversion_bit_for_bit() {
        let meta = paired();
        let encoded = encode(&meta);
        let beta_x = quantity("beta_x", QuantityRole::Covariate, "coefficient");
        let beta_y = quantity("beta_y", QuantityRole::Covariate, "coefficient");
        let expected = identity(
            DistributionMeaningWire::ParameterPosterior,
            &[beta_y.clone(), beta_x.clone()],
            "source-posterior",
        );
        let bindings =
            bind(&[meta.quantities[0].clone(), meta.quantities[1].clone()], &[beta_x, beta_y]);
        let facade = convert_legacy_posterior(&encoded, expected.clone(), &bindings).unwrap();
        let core = convert_core(&encoded, expected.clone(), &bindings).unwrap();
        assert_eq!(facade, core);
        assert_eq!(facade.draws(), &[0.0, 0.0, 2.0, 1.0, 4.0, 2.0]);
        assert_eq!(facade.semantic(), DistributionMeaningWire::ParameterPosterior);
        assert_eq!(facade.to_bytes("converted").unwrap(), core.to_bytes("converted").unwrap());
        let mut bytes = Vec::new();
        encoded.write_to(&mut bytes).unwrap();
        let reread = EncodedArtifact::read_from(bytes.as_slice()).unwrap();
        assert_eq!(
            convert_legacy_posterior(&reread, expected.clone(), &bindings).unwrap(),
            convert_core(&reread, expected.clone(), &bindings).unwrap()
        );
        let reversed: Vec<_> = bindings.iter().rev().cloned().collect();
        let reordered = convert_legacy_posterior(&encoded, expected.clone(), &reversed).unwrap();
        assert_eq!(reordered, convert_core(&encoded, expected, &reversed).unwrap());
        assert_eq!(reordered.draws(), facade.draws());
        assert_eq!(reordered.metadata().legacy_bindings.as_deref(), Some(reversed.as_slice()));
    }

    #[test]
    fn draw_meanings_stay_distinct_and_match_the_core_refusal() {
        let coefficients = paired();
        let mut effects = coefficients.clone();
        effects.quantities = vec![
            PosteriorQuantityWire::Effect { name: "ate_x".into() },
            PosteriorQuantityWire::Effect { name: "ate_y".into() },
        ];
        let cases = [
            (
                coefficients,
                quantity("beta_x", QuantityRole::Covariate, "coefficient"),
                quantity("beta_y", QuantityRole::Covariate, "coefficient"),
                DistributionMeaningWire::ParameterPosterior,
            ),
            (
                effects,
                quantity("ate_x", QuantityRole::Outcome, "mean"),
                quantity("ate_y", QuantityRole::Outcome, "mean"),
                DistributionMeaningWire::CausalFunctionalPosterior,
            ),
        ];
        for (meta, left, right, admitted) in cases {
            let encoded = encode(&meta);
            let bindings = bind(
                &[meta.quantities[0].clone(), meta.quantities[1].clone()],
                &[left.clone(), right.clone()],
            );
            for meaning in MEANINGS {
                let expected =
                    identity(meaning, &[right.clone(), left.clone()], "source-posterior");
                assert_same(&encoded, &expected, &bindings);
                let facade = convert_legacy_posterior(&encoded, expected, &bindings);
                if meaning == admitted {
                    assert_eq!(facade.unwrap().semantic(), admitted);
                } else {
                    assert_eq!(
                        facade.unwrap_err().reason_code(),
                        Some("distribution_meaning_mismatch")
                    );
                }
            }
        }
    }

    #[test]
    fn unnamed_outcomes_and_structural_mass_match_the_core_refusal() {
        let base = paired();
        let beta_x = quantity("beta_x", QuantityRole::Covariate, "coefficient");
        let beta_y = quantity("beta_y", QuantityRole::Covariate, "coefficient");
        let parameter = identity(
            DistributionMeaningWire::ParameterPosterior,
            &[beta_y.clone(), beta_x.clone()],
            "source-posterior",
        );
        let named =
            bind(&[base.quantities[0].clone(), base.quantities[1].clone()], &[beta_x, beta_y]);
        let mut unnamed = base.clone();
        unnamed.quantities[0] = PosteriorQuantityWire::Coefficient { index: 0, name: None };
        assert_same(&encode(&unnamed), &parameter, &named);

        let ate_x = quantity("ate_x", QuantityRole::Outcome, "mean");
        let ate_y = quantity("ate_y", QuantityRole::Outcome, "mean");
        let structural = identity(
            DistributionMeaningWire::CausalFunctionalPosterior,
            &[ate_y.clone(), ate_x.clone()],
            "source-posterior",
        );
        let effect_bindings = bind(
            &[
                PosteriorQuantityWire::Effect { name: "ate_x".into() },
                PosteriorQuantityWire::Effect { name: "ate_y".into() },
            ],
            &[ate_x, ate_y],
        );
        for name in ["", " "] {
            let mut blank = base.clone();
            blank.quantities = vec![
                PosteriorQuantityWire::Effect { name: name.into() },
                PosteriorQuantityWire::Effect { name: "ate_y".into() },
            ];
            assert_same(&encode(&blank), &structural, &effect_bindings);
            assert_eq!(
                convert_legacy_posterior(&encode(&blank), structural.clone(), &effect_bindings)
                    .unwrap_err()
                    .reason_code(),
                Some("distribution_meaning_mismatch")
            );
        }
        let mut scalars = base.clone();
        scalars.quantities = vec![
            PosteriorQuantityWire::Scalar { name: "ate_x".into() },
            PosteriorQuantityWire::Scalar { name: "ate_y".into() },
        ];
        assert_same(&encode(&scalars), &structural, &effect_bindings);

        let mut mass = base.clone();
        mass.quantities = vec![
            PosteriorQuantityWire::Effect { name: "ate_x".into() },
            PosteriorQuantityWire::Effect { name: "ate_y".into() },
        ];
        mass.unidentified_mass = 0.25;
        assert_same(&encode(&mass), &structural, &effect_bindings);
        let mut stopped = mass;
        stopped.unidentified_mass = 0.0;
        stopped.converged = false;
        assert_same(&encode(&stopped), &structural, &effect_bindings);

        let mut wrong_source = parameter.clone();
        wrong_source.source_id = "other-posterior".into();
        assert_same(&encode(&base), &wrong_source, &named);
    }
}
