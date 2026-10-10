//! 2.3A X4 remainder / B1 model-provider row: the learned joint transport artifact (model
//! with its polynomial basis, priors with provenance, data identities, exact posterior fitted
//! through `antecedent-learn`, aligned joint draws, diagnostics, graph/provider/query record,
//! identification, calibration `unmeasured`) replays independently.
//!
//! The oracle is closed-form Gauss-Jordan algebra on the posterior precision and never calls
//! the code under test. The public interval route stays closed; nothing here asserts coverage.
#![allow(
    clippy::cast_precision_loss,
    clippy::needless_range_loop,
    reason = "small dense closed-form algebra with short symbol names"
)]

use antecedent_core::{ExecutionContext, VariableId};
use antecedent_estimate::joint_bayesian_transport::{
    DataIdentity, GaussianPrior, JointPriors, JointTransportOptions, PriorProvenance, SourceData,
    SourceDependence, SourceSharing, TargetData, TransportGraphClass, VaryingBlock,
};
use antecedent_estimate::learned_joint_transport::{LearnedJointFit, LearnedJointModel};
use antecedent_identify::{
    PopulationFactor, TransportCertificate, TransportFormula, TransportIdentification,
};
use antecedent_io::IoError;
use antecedent_io::joint_bayesian_transport_artifact::{DataIdentityWire, PriorProvenanceWire};
use antecedent_io::learned_joint_transport_artifact::{
    LearnedJointArtifactWire, LearnedJointConsumeLimits, LearnedJointExpectation,
};
use std::sync::Arc;

const DRAWS: usize = 3000;
const FEATURE: u32 = 7;

fn source(id: &str, offset: usize, effect: f64) -> SourceData {
    let n = 40;
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
        y.push(
            1.0 + 0.5 * xi
                + 0.4 * xi * xi
                + t * (effect + 1.5 * xi + 1.2 * xi * xi)
                + 0.3 * (1.3 * k).cos(),
        );
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

fn identified() -> TransportIdentification {
    TransportIdentification::Transportable {
        formula: TransportFormula::Standardize {
            over: Arc::from([VariableId::from_raw(FEATURE)]),
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

fn model() -> LearnedJointModel {
    LearnedJointModel {
        graph: TransportGraphClass::FixedDag,
        features: vec![FEATURE],
        basis_degree: 2,
        varying: VaryingBlock::Intercept,
        sharing: SourceSharing::IndependentVaryingBlocks,
        dependence: SourceDependence::IndependentSamples,
        priors: JointPriors {
            invariant: GaussianPrior::isotropic(5, 0.0, 4.0, PriorProvenance::Declared),
            varying: GaussianPrior::isotropic(1, 0.0, 4.0, PriorProvenance::Declared),
        },
        max_unsupported_mass: 0.0,
        conflict_z_threshold: 3.0,
    }
}

fn target_x() -> Vec<f64> {
    (0..30_i32).map(|j| 0.4 * (0.21 * f64::from(j)).cos() + 0.3).collect()
}

fn target() -> TargetData {
    TargetData {
        identity: DataIdentity {
            snapshot_digest: "snap-target".into(),
            datum_ids: (0..30).map(|j| format!("t-{j}")).collect(),
        },
        rows: 30,
        covariates: vec![target_x()],
    }
}

fn sources() -> Vec<SourceData> {
    vec![source("s1", 0, 2.0), source("s2", 400, 2.0)]
}

fn options() -> JointTransportOptions {
    JointTransportOptions { draws: DRAWS, seed: 77 }
}

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(1)
}

fn built() -> (LearnedJointArtifactWire, LearnedJointFit) {
    LearnedJointArtifactWire::build(
        &identified(),
        &model(),
        &sources(),
        &target(),
        options(),
        &ctx(),
    )
    .expect("the fixture fits and exports")
}

fn consume(bytes: &[u8]) -> Result<(LearnedJointArtifactWire, LearnedJointFit), IoError> {
    LearnedJointArtifactWire::consume_with_limits(
        bytes,
        LearnedJointConsumeLimits::default(),
        &ctx(),
    )
}

fn reseal(wire: &mut LearnedJointArtifactWire) {
    wire.premises_digest = wire.expected_premises_digest().expect("premises digest");
    wire.data_digest = wire.expected_data_digest().expect("data digest");
}

/// Export `wire` after `mutate`, with both digests recomputed so only replay can refuse.
fn resealed(
    wire: &LearnedJointArtifactWire,
    mutate: impl FnOnce(&mut LearnedJointArtifactWire),
) -> Vec<u8> {
    let mut copy = wire.clone();
    mutate(&mut copy);
    reseal(&mut copy);
    copy.export().expect("export")
}

fn refusal(bytes: &[u8]) -> String {
    consume(bytes).expect_err("the consumer must refuse").to_string()
}

// ---------------------------------------------------------------- independent oracle

fn invert(mut a: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let n = a.len();
    let mut inv: Vec<Vec<f64>> =
        (0..n).map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect()).collect();
    for col in 0..n {
        let mut pivot = col;
        for r in col..n {
            if a[r][col].abs() > a[pivot][col].abs() {
                pivot = r;
            }
        }
        a.swap(col, pivot);
        inv.swap(col, pivot);
        let d = a[col][col];
        for j in 0..n {
            a[col][j] /= d;
            inv[col][j] /= d;
        }
        for r in 0..n {
            if r != col {
                let f = a[r][col];
                for j in 0..n {
                    a[r][j] -= f * a[col][j];
                    inv[r][j] -= f * inv[col][j];
                }
            }
        }
    }
    inv
}

/// Posterior mean and covariance of `[A, A x, A x^2, x, x^2, gamma_1.., gamma_S]` with every
/// prior `N(0, 4)`: precision `I/4 + sum Z'Z / sigma^2`.
fn oracle(sources: &[SourceData]) -> (Vec<f64>, Vec<Vec<f64>>) {
    let dim = 5 + sources.len();
    let mut precision = vec![vec![0.0; dim]; dim];
    for i in 0..dim {
        precision[i][i] = 0.25;
    }
    let mut moment = vec![0.0; dim];
    for (s, src) in sources.iter().enumerate() {
        let w = 1.0 / src.noise_variance;
        for i in 0..src.outcome.len() {
            let x = src.covariates[0][i];
            let a = if src.treatment[i] { 1.0 } else { 0.0 };
            let mut row = vec![0.0; dim];
            row[0] = a;
            row[1] = a * x;
            row[2] = a * x * x;
            row[3] = x;
            row[4] = x * x;
            row[5 + s] = 1.0;
            for j in 0..dim {
                moment[j] += w * row[j] * src.outcome[i];
                for k in 0..dim {
                    precision[j][k] += w * row[j] * row[k];
                }
            }
        }
    }
    let covariance = invert(precision);
    let mean = (0..dim).map(|j| (0..dim).map(|k| covariance[j][k] * moment[k]).sum()).collect();
    (mean, covariance)
}

// ---------------------------------------------------------------- tests

#[test]
fn a2_learned_artifact_replays_value_for_value_against_the_exact_oracle() {
    let (wire, fit) = built();
    let bytes = wire.export().expect("export");
    let (loaded, replayed) = consume(&bytes).expect("a faithful artifact replays");
    // The replay is the original fit, every moment and every aligned draw.
    assert_eq!(replayed, fit);
    assert_eq!(loaded.result.draws.values, fit.draws.values);
    assert_eq!(loaded.result.draws.n_draws, DRAWS);
    assert_eq!(loaded.result.draws.alignment, "joint");
    assert_eq!(loaded.calibration, "unmeasured");
    assert_eq!(loaded.result.diagnostics.calibration, "unmeasured");
    assert_eq!(loaded.result.identification_status, "identified");
    assert_eq!(loaded.result.identification_formula, "standardize");
    assert_ne!(loaded.premises_digest, loaded.data_digest);
    // The exact posterior equals the independent oracle.
    let (mean, covariance) = oracle(&sources());
    let dim = mean.len();
    assert_eq!(replayed.posterior_mean.len(), dim);
    for i in 0..dim {
        assert!((replayed.posterior_mean[i] - mean[i]).abs() < 1e-9, "mean {i}");
        for j in 0..dim {
            let stored = replayed.posterior_covariance[i * dim + j];
            assert!((stored - covariance[i][j]).abs() < 1e-9, "covariance {i},{j}");
        }
    }
    let tx = target_x();
    let xbar = tx.iter().sum::<f64>() / 30.0;
    let x2bar = tx.iter().map(|x| x * x).sum::<f64>() / 30.0;
    let c = [1.0, xbar, x2bar];
    let effect: f64 = (0..3).map(|i| c[i] * mean[i]).sum();
    let mut variance = 0.0;
    for i in 0..3 {
        for j in 0..3 {
            variance += c[i] * c[j] * covariance[i][j];
        }
    }
    assert!((loaded.result.target_effect_mean - effect).abs() < 1e-9);
    assert!((loaded.result.target_effect_variance - variance).abs() < 1e-9);
}

#[test]
fn a2_learned_artifact_records_the_learn_provider_row_and_basis_rank() {
    let (wire, fit) = built();
    let provider = &wire.result.provider;
    assert_eq!(provider.provider, "antecedent-learn");
    assert_eq!(provider.graph, "fixed_dag");
    assert_eq!(provider.query, "target_average_effect");
    assert_eq!(provider.basis_id, "polynomial_degree_2");
    assert_eq!(provider.basis_terms, ["x7", "x7^2"]);
    assert_eq!(provider.learn_model_id, "bayesian_basis.known_variance_gaussian");
    assert_eq!(wire.model.basis_degree, 2);
    assert_eq!(wire.result.diagnostics.basis_rank, fit.parameter_names.len());
    assert_eq!(wire.result.diagnostics.sampler, "learn_conjugate_gaussian_iid");
    assert!(wire.result.model_identity.starts_with("learned_joint_transport_v1|"));
}

#[test]
fn a2_learned_artifact_enforces_an_expected_identity() {
    let (wire, _) = built();
    let bytes = wire.export().expect("export");
    let expected = LearnedJointExpectation {
        premises_digest: Some(wire.premises_digest.clone()),
        data_digest: Some(wire.data_digest.clone()),
    };
    let accepted = LearnedJointArtifactWire::consume_expecting(
        &bytes,
        &expected,
        LearnedJointConsumeLimits::default(),
        &ctx(),
    );
    assert!(accepted.is_ok());
    // A resealed change of a source identity still replays numerically (the identity is a
    // declared premise), so only the consumer's expected identity refuses it.
    let changed = resealed(&wire, |w| w.sources[0].identity.snapshot_digest = "snap-other".into());
    assert!(consume(&changed).is_ok());
    let refused = LearnedJointArtifactWire::consume_expecting(
        &changed,
        &expected,
        LearnedJointConsumeLimits::default(),
        &ctx(),
    )
    .expect_err("a different data identity is refused")
    .to_string();
    assert!(refused.contains("differs from the consumer's expectation"), "{refused}");
    // The same holds for a changed premises identity.
    let prior = resealed(&wire, |w| w.model.conflict_z_threshold = 2.5);
    let refused = LearnedJointArtifactWire::consume_expecting(
        &prior,
        &expected,
        LearnedJointConsumeLimits::default(),
        &ctx(),
    )
    .expect_err("a different premises identity is refused")
    .to_string();
    assert!(refused.contains("differs from the consumer's expectation"), "{refused}");
}

type Mutation = (&'static str, fn(&mut LearnedJointArtifactWire));

#[test]
fn a2_learned_artifact_refuses_resealed_semantic_mutations() {
    let (wire, _) = built();
    let mutations: [Mutation; 13] = [
        ("prior mean", |w| w.model.invariant_prior.mean[0] = 0.75),
        ("prior covariance", |w| w.model.varying_prior.covariance[0] = 9.0),
        ("source outcome", |w| w.sources[0].outcome[3] += 0.5),
        ("noise variance", |w| w.sources[1].noise_variance = 0.2),
        ("target covariate", |w| w.target.covariates[0][0] += 0.01),
        ("draw count", |w| w.options.draws += 1),
        ("seed", |w| w.options.seed += 1),
        ("sharing", |w| w.model.sharing = "shared_varying_block".into()),
        ("rng id", |w| w.result.draws.rng_id = "splitmix64_box_muller_v1:seed=1:stream=0".into()),
        ("stored posterior", |w| w.result.posterior_mean[0] += 1e-9),
        ("provider basis", |w| w.result.provider.basis_id = "polynomial_degree_3".into()),
        ("provider name", |w| w.result.provider.provider = "antecedent-estimate".into()),
        ("basis rank", |w| w.result.diagnostics.basis_rank += 1),
    ];
    for (name, mutate) in mutations {
        let message = refusal(&resealed(&wire, mutate));
        assert!(message.contains("does not replay"), "{name}: {message}");
    }
    // A changed basis degree no longer matches the stored prior dimensions: refused by the
    // engine itself before any replay comparison.
    let message = refusal(&resealed(&wire, |w| w.model.basis_degree = 3));
    assert!(message.contains("learned_joint_transport.prior_dimension"), "{message}");
    let message = refusal(&resealed(&wire, |w| w.model.basis_degree = 0));
    assert!(message.contains("learned_joint_transport.invalid_basis"), "{message}");
    // A changed covariate declaration is refused by the engine itself.
    let message = refusal(&resealed(&wire, |w| w.model.features = vec![8]));
    assert!(message.contains("learned_joint_transport.feature_mismatch"), "{message}");
    // A changed graph class is refused as unsupported before any fit.
    let message = refusal(&resealed(&wire, |w| w.model.graph = "admg".into()));
    assert!(message.contains("learned_joint_transport.unsupported_graph"), "{message}");
    // Undeclared source dependence is refused.
    let message = refusal(&resealed(&wire, |w| w.model.dependence = "unknown".into()));
    assert!(message.contains("learned_joint_transport.source_dependence"), "{message}");
    // Without resealing, an edited observation breaks the data identity digest.
    let mut edited = wire.clone();
    edited.sources[0].outcome[0] += 1.0;
    let message = refusal(&edited.export().expect("export"));
    assert!(message.contains("data identity digest mismatch"), "{message}");
    // Without resealing, an edited prior breaks the premises digest.
    let mut edited = wire.clone();
    edited.model.invariant_prior.mean[1] = 0.3;
    let message = refusal(&edited.export().expect("export"));
    assert!(message.contains("premises digest mismatch"), "{message}");
    // Without resealing, an edited basis degree breaks the premises digest.
    let mut edited = wire.clone();
    edited.model.basis_degree = 1;
    let message = refusal(&edited.export().expect("export"));
    assert!(message.contains("premises digest mismatch"), "{message}");
}

#[test]
fn a2_learned_artifact_refuses_prior_bank_and_likelihood_double_use() {
    let overlapping = PriorProvenance::Bank {
        bank_id: "bank-1".into(),
        consumed: vec![DataIdentity { snapshot_digest: "snap-s1".into(), datum_ids: vec![] }],
    };
    let mut double = model();
    double.priors.varying.provenance = overlapping;
    let produced = LearnedJointArtifactWire::build(
        &identified(),
        &double,
        &sources(),
        &target(),
        options(),
        &ctx(),
    )
    .expect_err("the producer refuses double use");
    assert_eq!(produced.reason_code(), Some("invalid_argument"));
    assert!(produced.to_string().contains("learned_joint_transport.prior_data_overlap"));
    // A consumer given a resealed artifact whose prior bank consumed a source refuses too.
    let (wire, _) = built();
    let message = refusal(&resealed(&wire, |w| {
        w.model.varying_prior.provenance = PriorProvenanceWire::Bank {
            bank_id: "bank-1".into(),
            consumed: vec![DataIdentityWire {
                snapshot_digest: "snap-s2".into(),
                datum_ids: vec![],
            }],
        };
    }));
    assert!(message.contains("learned_joint_transport.prior_data_overlap"), "{message}");
    // A bank built from other observations is accepted and stays recorded.
    let mut disjoint = model();
    disjoint.priors.varying.provenance = PriorProvenance::Bank {
        bank_id: "bank-2".into(),
        consumed: vec![DataIdentity { snapshot_digest: "snap-historic".into(), datum_ids: vec![] }],
    };
    let (banked, _) = LearnedJointArtifactWire::build(
        &identified(),
        &disjoint,
        &sources(),
        &target(),
        options(),
        &ctx(),
    )
    .expect("a disjoint bank fits");
    assert!(consume(&banked.export().expect("export")).is_ok());
    assert!(matches!(
        banked.model.varying_prior.provenance,
        PriorProvenanceWire::Bank { ref bank_id, .. } if bank_id == "bank-2"
    ));
}

#[test]
fn a2_learned_artifact_refuses_an_unknown_version_foreign_feature_and_claim() {
    let (wire, _) = built();
    let mut future = wire.clone();
    future.version = 7;
    let bytes = future.export().expect("export");
    assert!(matches!(
        LearnedJointArtifactWire::decode(&bytes),
        Err(IoError::UnsupportedVersion { version: 7 })
    ));
    let mut foreign = wire.clone();
    foreign.required_features = vec!["joint_bayesian_transport_conjugate_v1".into()];
    assert!(LearnedJointArtifactWire::decode(&foreign.export().expect("export")).is_err());
    let mut calibrated = wire.clone();
    calibrated.calibration = "calibrated".into();
    assert!(LearnedJointArtifactWire::decode(&calibrated.export().expect("export")).is_err());
    let mut marginals = wire;
    marginals.result.draws.alignment = "independent_marginals".into();
    assert!(LearnedJointArtifactWire::decode(&marginals.export().expect("export")).is_err());
}

#[test]
fn a2_learned_artifact_refuses_stored_sizes_above_the_consumer_limits() {
    let (wire, _) = built();
    let bytes = wire.export().expect("export");
    let tight = [
        LearnedJointConsumeLimits { max_draws: 100, ..LearnedJointConsumeLimits::default() },
        LearnedJointConsumeLimits { max_source_rows: 10, ..LearnedJointConsumeLimits::default() },
        LearnedJointConsumeLimits { max_draw_cells: 100, ..LearnedJointConsumeLimits::default() },
        LearnedJointConsumeLimits { max_sources: 1, ..LearnedJointConsumeLimits::default() },
        LearnedJointConsumeLimits { max_features: 0, ..LearnedJointConsumeLimits::default() },
    ];
    for limits in tight {
        let message = LearnedJointArtifactWire::consume_with_limits(&bytes, limits, &ctx())
            .expect_err("a stored size above the consumer limit is refused")
            .to_string();
        assert!(message.contains("consumer limit exceeded"), "{message}");
    }
}
