//! F19: exact heterogeneous-design Gaussian transfer and independent replay.

use antecedent::inference::mapped_posterior_transfer;
use antecedent_core::{QuantityRole, ScientificQuantity};
use antecedent_io::distribution_artifact::{
    DistributionArtifact, DistributionCalibration, DistributionIdentity, DistributionMetadata,
    DistributionProvenance, DistributionTrust, DrawAlignment,
};
use antecedent_io::quantity_wire::{DistributionMeaningWire, ScientificQuantityWire};
use antecedent_prob::{
    BayesDesignRef, BayesFitOptions, GaussianCoefficientPrior, LaplaceWorkspace, PriorSet,
    PriorSpec, fit_conjugate_gaussian,
};

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

fn identity() -> DistributionIdentity {
    DistributionIdentity::new(
        DistributionMeaningWire::ParameterPosterior,
        &[
            quantity("source:x", "coefficient"),
            quantity("source:y", "coefficient"),
            quantity("source:sigma2", "residual_variance"),
        ],
        DrawAlignment::Joint,
        DistributionProvenance {
            source_id: "batch-a".into(),
            provider_id: "conjugate-gaussian".into(),
            rng_id: "deterministic-exact-quadrature".into(),
            snapshot_id: "batch-a-snapshot".into(),
            causal_contract_id: "fixed-gaussian-contract".into(),
        },
    )
    .unwrap()
}

fn targets() -> Vec<ScientificQuantityWire> {
    [quantity("target:y", "coefficient"), quantity("target:x", "coefficient")]
        .iter()
        .map(ScientificQuantityWire::from)
        .collect()
}

fn baseline(sigma2: f64) -> PriorSet {
    let mut prior = PriorSet::new();
    prior.push(PriorSpec::GaussianCoefficients(GaussianCoefficientPrior::isotropic(2, 1.0)));
    prior.push(PriorSpec::KnownResidualVariance(sigma2));
    prior
}

fn exact_source() -> DistributionArtifact {
    // Batch A: x=(1,1), y=1, prior N(0,I), sigma²=1. Exact posterior:
    // mean=(1/3,1/3), covariance=[[2,-1],[-1,2]]/3.
    // Four symmetric quadrature draws have precisely those population moments.
    let mx = 1.0 / 3.0;
    let my = 1.0 / 3.0;
    let a = (4.0_f64 / 3.0).sqrt();
    let b = -(1.0_f64 / 3.0).sqrt();
    let c = 1.0_f64;
    let draws = vec![mx + a, my + b, 1.0, mx - a, my - b, 1.0, mx, my + c, 1.0, mx, my - c, 1.0];
    DistributionArtifact::new(
        DistributionMetadata {
            version: 1,
            identity: identity(),
            axes: ["draw".into(), "quantity".into()],
            shape: [4, 3],
            weights: None,
            supported: None,
            calibration: DistributionCalibration::PointOnly,
            trust: DistributionTrust::Unverified,
            legacy_posterior: None,
            legacy_bindings: None,
        },
        draws,
    )
    .unwrap()
}

fn mapped(
    bytes: &[u8],
    expected: &DistributionIdentity,
    prior: &PriorSet,
) -> Result<PriorSet, antecedent::error::CausalError> {
    mapped_posterior_transfer(
        bytes,
        expected,
        &[("source:x".into(), "target:x".into()), ("source:y".into(), "target:y".into())],
        &targets(),
        "batch-b-snapshot",
        "fixed-gaussian-contract",
        prior,
    )
}

#[test]
fn heterogeneous_design_sequential_equals_exact_pooled() {
    let bytes = exact_source().to_bytes("batch-a-posterior").unwrap();
    let prior = mapped(&bytes, &identity(), &baseline(1.0)).unwrap();
    assert!(prior.restrictions.iter().any(|receipt| {
        receipt.id.as_ref() == "mapped_posterior_transfer"
            && receipt.description.contains("batch-a-snapshot")
            && receipt.description.contains("batch-b-snapshot")
    }));
    let coef = prior.gaussian_coefficients().unwrap();
    assert!((coef.mean[0] - 1.0 / 3.0).abs() < 1e-12);
    assert!((coef.mean[1] - 1.0 / 3.0).abs() < 1e-12);
    assert!(prior.coefficient_correlation().is_some());
    let options = BayesFitOptions { n_draws: 32, ..BayesFitOptions::default() };
    // Batch B is x=(1,-1) in source order, hence (-1,1) in target order.
    let sequential = fit_conjugate_gaussian(
        BayesDesignRef {
            x_colmajor: &[-1.0, 1.0],
            nrows: 1,
            ncols: 2,
            y: &[3.0],
            weights: None,
            offsets: None,
        },
        &prior,
        &options,
        &mut LaplaceWorkspace::default(),
    )
    .unwrap();
    let pooled = fit_conjugate_gaussian(
        BayesDesignRef {
            x_colmajor: &[1.0, -1.0, 1.0, 1.0],
            nrows: 2,
            ncols: 2,
            y: &[1.0, 3.0],
            weights: None,
            offsets: None,
        },
        &baseline(1.0),
        &options,
        &mut LaplaceWorkspace::default(),
    )
    .unwrap();
    for i in 0..2 {
        assert!((sequential.map[i] - pooled.map[i]).abs() < 1e-12);
    }
    assert!((sequential.map[0] + 2.0 / 3.0).abs() < 1e-12);
    assert!((sequential.map[1] - 4.0 / 3.0).abs() < 1e-12);
    for i in 0..4 {
        assert!(
            (sequential.cov.as_ref().unwrap()[i] - pooled.cov.as_ref().unwrap()[i]).abs() < 1e-12
        );
    }
    assert!((sequential.cov.as_ref().unwrap()[0] - 1.0 / 3.0).abs() < 1e-12);
}

#[test]
fn incompatible_sources_and_mapping_refuse() {
    let artifact = exact_source();
    let bytes = artifact.to_bytes("batch-a-posterior").unwrap();
    assert!(mapped(&bytes, &identity(), &baseline(2.0)).is_err());
    let mut wrong_snapshot = identity();
    wrong_snapshot.snapshot_id = "other".into();
    assert!(mapped(&bytes, &wrong_snapshot, &baseline(1.0)).is_err());
    let mut bootstrap = artifact.metadata().clone();
    bootstrap.identity.semantic = DistributionMeaningWire::Bootstrap;
    let bootstrap = DistributionArtifact::new(bootstrap, artifact.draws().to_vec()).unwrap();
    assert!(
        mapped(
            &bootstrap.to_bytes("bootstrap").unwrap(),
            &bootstrap.metadata().identity,
            &baseline(1.0)
        )
        .is_err()
    );
    let reused = [("source:x".into(), "target:x".into()), ("source:x".into(), "target:y".into())];
    assert!(
        mapped_posterior_transfer(
            &bytes,
            &identity(),
            &reused,
            &targets(),
            "batch-b-snapshot",
            "fixed-gaussian-contract",
            &baseline(1.0)
        )
        .is_err()
    );
    let partial = [("source:x".into(), "target:x".into())];
    assert!(
        mapped_posterior_transfer(
            &bytes,
            &identity(),
            &partial,
            &targets(),
            "batch-b-snapshot",
            "fixed-gaussian-contract",
            &baseline(1.0)
        )
        .is_err()
    );
    assert!(
        mapped_posterior_transfer(
            &bytes,
            &identity(),
            &[("source:x".into(), "target:x".into()), ("source:y".into(), "target:y".into())],
            &targets(),
            "",
            "fixed-gaussian-contract",
            &baseline(1.0),
        )
        .is_err()
    );
    let mut changed_target = targets();
    changed_target[0].units = "meters".into();
    assert!(
        mapped_posterior_transfer(
            &bytes,
            &identity(),
            &[("source:x".into(), "target:x".into()), ("source:y".into(), "target:y".into())],
            &changed_target,
            "batch-b-snapshot",
            "fixed-gaussian-contract",
            &baseline(1.0)
        )
        .is_err()
    );
}

#[test]
fn fresh_process_replays_mapped_prior() {
    const PATH: &str = "ANTECEDENT_F19_REPLAY_PATH";
    if let Ok(path) = std::env::var(PATH) {
        let bytes = std::fs::read(path).unwrap();
        let prior = mapped(&bytes, &identity(), &baseline(1.0)).unwrap();
        assert!(prior.coefficient_correlation().is_some());
        let mut wrong = identity();
        wrong.snapshot_id = "resealed-other".into();
        assert!(mapped(&bytes, &wrong, &baseline(1.0)).is_err());
        return;
    }
    let unique =
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let path =
        std::env::temp_dir().join(format!("antecedent-f19-{}-{unique}.bin", std::process::id()));
    std::fs::write(&path, exact_source().to_bytes("batch-a-posterior").unwrap()).unwrap();
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .arg("--exact")
        .arg("fresh_process_replays_mapped_prior")
        .env(PATH, &path)
        .output()
        .unwrap();
    std::fs::remove_file(path).unwrap();
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
}
