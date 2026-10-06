//! 2.3A X4 `x4_posterior_hydration`: joint source-target transport draws become a portable
//! aligned joint distribution artifact with scientific quantity coordinates.
//!
//! Dependency direction verified: `antecedent-io` depends on `antecedent-estimate`
//! (its Cargo.toml), so this test lives here rather than in `antecedent-estimate`, which
//! cannot depend on `antecedent-io`. Calibration is unmeasured and the artifact says so.
#![allow(clippy::cast_precision_loss, reason = "small deterministic fixtures")]

use antecedent_core::{ExecutionContext, QuantityRole, ScientificQuantity, VariableId};
use antecedent_estimate::joint_bayesian_transport::{
    DataIdentity, GaussianPrior, JointPriors, JointTransportFit, JointTransportModel,
    JointTransportOptions, PriorProvenance, SourceData, SourceDependence, SourceSharing,
    TargetData, TransportGraphClass, VaryingBlock, fit_joint_bayesian_transport,
};
use antecedent_identify::{
    PopulationFactor, TransportCertificate, TransportFormula, TransportIdentification,
};
use antecedent_io::distribution_artifact::{
    DistributionArtifact, DistributionCalibration, DistributionIdentity, DistributionMetadata,
    DistributionProvenance, DistributionTrust, DrawAlignment,
};
use antecedent_io::quantity_wire::DistributionMeaningWire;
use std::sync::Arc;

const DRAWS: usize = 40_000;

fn source(id: &str, offset: usize, effect: f64) -> SourceData {
    let n = 50;
    let mut x = Vec::new();
    let mut a = Vec::new();
    let mut y = Vec::new();
    for i in 0..n {
        let k = (i + offset) as f64;
        let xi = 1.5 * (0.37 * k).sin() + 0.2;
        let treated = i % 2 == 0;
        let t = if treated { 1.0 } else { 0.0 };
        x.push(xi);
        a.push(treated);
        y.push(1.0 + 0.5 * xi + t * (effect + 1.5 * xi) + 0.3 * (1.3 * k).cos());
    }
    SourceData {
        id: id.into(),
        identity: DataIdentity {
            snapshot_digest: format!("snap-{id}"),
            datum_ids: (0..n).map(|i| format!("{id}-{i}")).collect(),
        },
        treatment: a,
        outcome: y,
        covariates: vec![x],
        noise_variance: 0.09,
    }
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

fn run() -> JointTransportFit {
    let id = TransportIdentification::Transportable {
        formula: TransportFormula::Standardize {
            over: Arc::from([VariableId::from_raw(7)]),
            source_response: factor(),
            target_law: factor(),
        },
        certificate: TransportCertificate {
            rule: Arc::from("standardization"),
            selection_targets: Arc::from([]),
            premises: Arc::from([]),
        },
    };
    let model = JointTransportModel {
        graph: TransportGraphClass::FixedDag,
        features: vec![7],
        varying: VaryingBlock::Intercept,
        sharing: SourceSharing::IndependentVaryingBlocks,
        dependence: SourceDependence::IndependentSamples,
        priors: JointPriors {
            invariant: GaussianPrior::isotropic(3, 0.0, 4.0, PriorProvenance::Declared),
            varying: GaussianPrior::isotropic(1, 0.0, 4.0, PriorProvenance::Declared),
        },
        max_unsupported_mass: 0.0,
        conflict_z_threshold: 3.0,
    };
    let target = TargetData {
        identity: DataIdentity {
            snapshot_digest: "snap-target".into(),
            datum_ids: (0..30).map(|j| format!("t-{j}")).collect(),
        },
        rows: 30,
        covariates: vec![(0..30_i32).map(|j| 0.4 * (0.21 * f64::from(j)).cos() + 0.3).collect()],
    };
    let sources = [source("s1", 0, 2.0), source("s2", 400, 2.0)];
    let ctx = ExecutionContext::for_tests(1);
    fit_joint_bayesian_transport(
        &id,
        &model,
        &sources,
        Some(&target),
        &JointTransportOptions { draws: DRAWS, seed: 77 },
        &ctx,
    )
    .unwrap()
}

fn effect_quantity(name: &str, population: &str) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: format!("schema:{name}"),
        variable_name: name.into(),
        role: QuantityRole::Outcome,
        units: "outcome_units".into(),
        population_id: population.into(),
        regime_id: "do(a=1)-do(a=0)".into(),
        horizon: 0,
        functional_id: "average_treatment_effect".into(),
        conditioning: vec![],
        transform_id: "identity".into(),
    }
}

fn identity(fit: &JointTransportFit, alignment: DrawAlignment) -> DistributionIdentity {
    let quantities = [
        effect_quantity("effect.source.s1", "source:s1"),
        effect_quantity("effect.source.s2", "source:s2"),
        effect_quantity("effect.target", "target"),
    ];
    DistributionIdentity::new(
        DistributionMeaningWire::CausalFunctionalPosterior,
        &quantities,
        alignment,
        DistributionProvenance {
            source_id: "trials:s1+s2".into(),
            provider_id: "antecedent.transport.joint_bayesian".into(),
            rng_id: fit.draws.rng_id.clone(),
            snapshot_id: "snap-s1+snap-s2+snap-target".into(),
            causal_contract_id: format!("{}:{}", fit.identification.rule, fit.model_identity),
        },
    )
    .unwrap()
}

fn artifact(fit: &JointTransportFit, alignment: DrawAlignment) -> DistributionArtifact {
    let (names, values) = fit.effect_draws();
    assert_eq!(names, ["effect.source.s1", "effect.source.s2", "effect.target"]);
    DistributionArtifact::new(
        DistributionMetadata {
            version: 1,
            identity: identity(fit, alignment),
            axes: ["draw".into(), "quantity".into()],
            shape: [fit.draws.n_draws, names.len()],
            weights: None,
            supported: None,
            calibration: DistributionCalibration::Unmeasured,
            trust: DistributionTrust::Unverified,
            legacy_posterior: None,
            legacy_bindings: None,
        },
        values,
    )
    .unwrap()
}

#[test]
fn x4_posterior_hydration_aligned_joint_draws_round_trip_with_expected_identity() {
    let fit = run();
    let original = artifact(&fit, DrawAlignment::Joint);
    let bytes = original.to_bytes("joint-bayesian-transport-fixture").unwrap();
    // A fresh consumer rebuilds the identity from its own contract, not from the artifact.
    let loaded = DistributionArtifact::from_bytes(&bytes, &identity(&fit, DrawAlignment::Joint))
        .unwrap();
    assert_eq!(loaded.shape(), [DRAWS, 3]);
    assert_eq!(loaded.metadata().calibration, DistributionCalibration::Unmeasured);
    assert_eq!(loaded.metadata().identity.alignment, DrawAlignment::Joint);
    assert_eq!(loaded.draws(), original.draws());
    let e = fit.effect_names.len();
    for i in 0..e {
        let sd = fit.effect_covariance[i * e + i].sqrt();
        assert!((loaded.mean(i).unwrap() - fit.effect_means[i]).abs() < 5.0 * sd / (DRAWS as f64).sqrt());
        for j in i..e {
            let analytic = fit.effect_covariance[i * e + j];
            let tol = 6.0
                * ((analytic * analytic
                    + fit.effect_covariance[i * e + i] * fit.effect_covariance[j * e + j])
                    / DRAWS as f64)
                    .sqrt();
            assert!((loaded.covariance(i, j).unwrap() - analytic).abs() < tol, "cov {i},{j}");
        }
    }
    // Source and target effects share theta, so their covariance is nonzero and the
    // nonlinear joint expectation uses the pairing: E[s t] = Cov + E[s] E[t].
    let cov = loaded.covariance(0, 2).unwrap();
    assert!(cov.abs() > 1e-6);
    let product = loaded.joint_expectation(0, 2, |s, t| s * t).unwrap();
    let expected = cov + loaded.mean(0).unwrap() * loaded.mean(2).unwrap();
    assert!((product - expected).abs() < 1e-9);
}

#[test]
fn x4_posterior_hydration_refuses_changed_meaning_and_independent_marginals() {
    let fit = run();
    let bytes = artifact(&fit, DrawAlignment::Joint).to_bytes("fixture").unwrap();
    // The consumer expects a different target population: refused.
    let mut wrong = identity(&fit, DrawAlignment::Joint);
    wrong.quantities[2].population_id = "another_target".into();
    assert!(DistributionArtifact::from_bytes(&bytes, &wrong).is_err());
    // The consumer expects a different model contract: refused.
    let mut other_model = identity(&fit, DrawAlignment::Joint);
    other_model.causal_contract_id = "another-model".into();
    assert!(DistributionArtifact::from_bytes(&bytes, &other_model).is_err());
    // The same draws labelled as independent marginals carry no covariance.
    let marginal = artifact(&fit, DrawAlignment::IndependentMarginals);
    assert!(marginal.covariance(0, 2).is_err());
    assert!(marginal.joint_expectation(0, 2, |s, t| s * t).is_err());
    let bytes = marginal.to_bytes("marginals").unwrap();
    assert!(DistributionArtifact::from_bytes(&bytes, &identity(&fit, DrawAlignment::Joint)).is_err());
}
