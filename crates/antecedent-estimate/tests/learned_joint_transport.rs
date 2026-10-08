//! 2.3A X4 remainder / B1 model-provider row: learned joint source-target transport whose
//! outcome mechanism is fitted through `antecedent-learn`'s Bayesian basis regression.
//!
//! The oracle below is written out in closed-form matrix algebra (Gauss-Jordan inverse of the
//! posterior precision, polynomial basis written out by hand) and never calls the code under
//! test. The one exception is `a2_learned_linear_basis_reproduces_joint_bayesian_moments`,
//! which is a labelled regression cross-check between two Antecedent paths, not independent
//! truth. Calibration is unmeasured; nothing here asserts coverage.
#![allow(
    clippy::cast_precision_loss,
    clippy::similar_names,
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    reason = "small dense closed-form algebra with short symbol names"
)]

use antecedent_core::{ExecutionContext, VariableId};
use antecedent_estimate::EstimationError;
use antecedent_estimate::joint_bayesian_transport::{
    DataIdentity, GaussianPrior, JointDraws, JointPriors, JointTransportCalibration,
    JointTransportModel, JointTransportOptions, JointTransportRefusal, PriorProvenance, SourceData,
    SourceDependence, SourceSharing, TargetData, TransportGraphClass, VaryingBlock,
    fit_joint_bayesian_transport,
};
use antecedent_estimate::learned_joint_transport::{
    DETAIL_CANCELLED, DETAIL_FEATURE_MISMATCH, DETAIL_FIT_FAILED, DETAIL_IDENTIFICATION,
    DETAIL_INVALID_BASIS, DETAIL_INVALID_INPUT, DETAIL_INVALID_PRIOR, DETAIL_MISSING_LAW,
    DETAIL_MISSING_SOURCE, DETAIL_PRIOR_DATA_OVERLAP, DETAIL_PRIOR_DIMENSION,
    DETAIL_RANK_DEFICIENT, DETAIL_ROUTE_FROZEN, DETAIL_SOURCE_DEPENDENCE, DETAIL_TOO_MANY_DRAWS,
    DETAIL_TOO_MANY_PARAMETERS, DETAIL_UNSUPPORTED_GRAPH, DETAIL_WEAK_OVERLAP,
    LEARNED_JOINT_PROVIDER, LEARNED_JOINT_QUERY, LEARNED_JOINT_SAMPLER, LearnedJointFit,
    LearnedJointModel, fit_learned_joint_transport, route_frozen_refusal,
};
use antecedent_identify::{
    MissingEvidenceCertificate, NotCertifiedCertificate, PopulationFactor, TransportCertificate,
    TransportFormula, TransportIdentification,
};
use std::sync::Arc;

const DRAWS: usize = 40_000;
const FEATURE: u32 = 7;

// ---------------------------------------------------------------- fixtures

struct Raw {
    x: Vec<f64>,
    a: Vec<bool>,
    y: Vec<f64>,
    noise: f64,
}

/// Quadratic truth `E[Y|a,x] = b0 + 0.5 x + 0.4 x^2 + a (effect + slope x + curve x^2)` with a
/// bounded deterministic perturbation of size `wiggle`.
fn raw(
    n: usize,
    offset: usize,
    b0: f64,
    effect: f64,
    (slope, curve): (f64, f64),
    (noise, wiggle): (f64, f64),
) -> Raw {
    let mut r = Raw { x: vec![], a: vec![], y: vec![], noise };
    for i in 0..n {
        let k = (i + offset) as f64;
        let x = 1.5 * (0.37 * k).sin() + 0.2;
        let a = i % 2 == 0;
        let t = if a { 1.0 } else { 0.0 };
        let y = b0
            + 0.5 * x
            + 0.4 * x * x
            + t * (effect + slope * x + curve * x * x)
            + wiggle * (1.3 * k).cos();
        r.x.push(x);
        r.a.push(a);
        r.y.push(y);
    }
    r
}

fn to_source(id: &str, r: &Raw, digest: &str, prefix: &str) -> SourceData {
    SourceData {
        id: id.into(),
        identity: DataIdentity {
            snapshot_digest: digest.into(),
            datum_ids: (0..r.y.len()).map(|i| format!("{prefix}-{i}")).collect(),
        },
        treatment: r.a.clone(),
        outcome: r.y.clone(),
        covariates: vec![r.x.clone()],
        noise_variance: r.noise,
    }
}

fn target_x() -> Vec<f64> {
    (0..40_i32).map(|j| 0.4 * (0.21 * f64::from(j)).cos() + 0.3).collect()
}

fn target() -> TargetData {
    TargetData {
        identity: DataIdentity {
            snapshot_digest: "target-snapshot".into(),
            datum_ids: (0..40).map(|j| format!("t-{j}")).collect(),
        },
        rows: 40,
        covariates: vec![target_x()],
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

fn certificate() -> TransportCertificate {
    TransportCertificate {
        rule: Arc::from("standardization"),
        selection_targets: Arc::from([]),
        premises: Arc::from([]),
    }
}

fn identified() -> TransportIdentification {
    TransportIdentification::Transportable {
        formula: TransportFormula::Standardize {
            over: Arc::from([VariableId::from_raw(FEATURE)]),
            source_response: factor(),
            target_law: factor(),
        },
        certificate: certificate(),
    }
}

fn dims(degree: usize, varying: VaryingBlock) -> (usize, usize) {
    match varying {
        VaryingBlock::Intercept => (1 + 2 * degree, 1),
        VaryingBlock::InterceptAndCovariates => (1 + degree, 1 + degree),
    }
}

fn learned_model(
    degree: usize,
    varying: VaryingBlock,
    sharing: SourceSharing,
    theta_mean: f64,
    theta_var: f64,
) -> LearnedJointModel {
    let (q, r) = dims(degree, varying);
    LearnedJointModel {
        graph: TransportGraphClass::FixedDag,
        features: vec![FEATURE],
        basis_degree: degree,
        varying,
        sharing,
        dependence: SourceDependence::IndependentSamples,
        priors: JointPriors {
            invariant: GaussianPrior::isotropic(
                q,
                theta_mean,
                theta_var,
                PriorProvenance::Declared,
            ),
            varying: GaussianPrior::isotropic(r, 0.0, 4.0, PriorProvenance::Declared),
        },
        max_unsupported_mass: 0.0,
        conflict_z_threshold: 3.0,
    }
}

fn model() -> LearnedJointModel {
    learned_model(2, VaryingBlock::Intercept, SourceSharing::IndependentVaryingBlocks, 0.0, 4.0)
}

fn options() -> JointTransportOptions {
    JointTransportOptions { draws: DRAWS, seed: 20_231_104 }
}

fn small() -> JointTransportOptions {
    JointTransportOptions { draws: 50, seed: 1 }
}

// The public refusal type is deliberately rich; tests exercise it unboxed.
#[allow(clippy::result_large_err)]
fn fit(
    model: &LearnedJointModel,
    sources: &[SourceData],
    options: &JointTransportOptions,
) -> Result<LearnedJointFit, JointTransportRefusal> {
    let ctx = ExecutionContext::for_tests(1);
    fit_learned_joint_transport(&identified(), model, sources, Some(&target()), options, &ctx)
}

fn refusal_of(
    model: &LearnedJointModel,
    sources: &[SourceData],
    target: Option<&TargetData>,
) -> JointTransportRefusal {
    let ctx = ExecutionContext::for_tests(1);
    fit_learned_joint_transport(&identified(), model, sources, target, &small(), &ctx)
        .expect_err("fit must refuse")
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

/// Polynomial basis `x, x^2, .., x^degree`, written out by hand.
fn phi(x: f64, degree: usize) -> Vec<f64> {
    (1..=degree).map(|k| (0..k).fold(1.0, |acc, _| acc * x)).collect()
}

struct Oracle {
    mean: Vec<f64>,
    cov: Vec<Vec<f64>>,
    dim: usize,
}

#[derive(Clone)]
struct Spec<'a> {
    degree: usize,
    vc: bool,
    shared: bool,
    theta_mean: Vec<f64>,
    theta_cov: Vec<Vec<f64>>,
    gamma_var: f64,
    raws: &'a [&'a Raw],
}

fn iso(dim: usize, var: f64) -> Vec<Vec<f64>> {
    (0..dim).map(|i| (0..dim).map(|j| if i == j { var } else { 0.0 }).collect()).collect()
}

fn oracle(spec: &Spec<'_>) -> Oracle {
    let m = spec.degree;
    let (q, r) = if spec.vc { (1 + m, 1 + m) } else { (1 + 2 * m, 1) };
    let blocks = if spec.shared { 1 } else { spec.raws.len() };
    let dim = q + blocks * r;
    let mut precision = vec![vec![0.0; dim]; dim];
    let mut rhs = vec![0.0; dim];
    let theta_precision = invert(spec.theta_cov.clone());
    for i in 0..q {
        for j in 0..q {
            precision[i][j] = theta_precision[i][j];
            rhs[i] += theta_precision[i][j] * spec.theta_mean[j];
        }
    }
    for i in q..dim {
        precision[i][i] = 1.0 / spec.gamma_var;
    }
    for (s, rw) in spec.raws.iter().enumerate() {
        let off = q + if spec.shared { 0 } else { s } * r;
        for i in 0..rw.y.len() {
            let a = if rw.a[i] { 1.0 } else { 0.0 };
            let p = phi(rw.x[i], m);
            let mut z = vec![0.0; dim];
            z[0] = a;
            for k in 0..m {
                z[1 + k] = a * p[k];
                if !spec.vc {
                    z[1 + m + k] = p[k];
                }
            }
            z[off] = 1.0;
            if spec.vc {
                for k in 0..m {
                    z[off + 1 + k] = p[k];
                }
            }
            for j in 0..dim {
                rhs[j] += z[j] * rw.y[i] / rw.noise;
                for k in 0..dim {
                    precision[j][k] += z[j] * z[k] / rw.noise;
                }
            }
        }
    }
    let cov = invert(precision);
    let mean = (0..dim).map(|i| (0..dim).map(|j| cov[i][j] * rhs[j]).sum()).collect();
    Oracle { mean, cov, dim }
}

fn phibar(xs: &[f64], degree: usize) -> Vec<f64> {
    let mut total = vec![0.0; degree];
    for x in xs {
        for (k, v) in phi(*x, degree).iter().enumerate() {
            total[k] += v;
        }
    }
    total.iter().map(|t| t / xs.len() as f64).collect()
}

fn effect_vec(dim: usize, degree: usize, xs: &[f64]) -> Vec<f64> {
    let mut c = vec![0.0; dim];
    c[0] = 1.0;
    c[1..=degree].copy_from_slice(&phibar(xs, degree));
    c
}

fn quad(cov: &[Vec<f64>], a: &[f64], b: &[f64]) -> f64 {
    let mut s = 0.0;
    for i in 0..a.len() {
        for j in 0..b.len() {
            s += a[i] * cov[i][j] * b[j];
        }
    }
    s
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Effect vectors: each source then the target.
fn effect_vectors(o: &Oracle, degree: usize, raws: &[&Raw]) -> Vec<Vec<f64>> {
    let mut v: Vec<Vec<f64>> = raws.iter().map(|r| effect_vec(o.dim, degree, &r.x)).collect();
    v.push(effect_vec(o.dim, degree, &target_x()));
    v
}

fn assert_matches_oracle(f: &LearnedJointFit, o: &Oracle, degree: usize, raws: &[&Raw]) {
    let d = o.dim;
    assert_eq!(f.posterior_mean.len(), d);
    for i in 0..d {
        assert!((f.posterior_mean[i] - o.mean[i]).abs() < 1e-9, "mean {i}");
        for j in 0..d {
            assert!(
                (f.posterior_covariance[i * d + j] - o.cov[i][j]).abs() < 1e-9,
                "covariance {i},{j}"
            );
        }
    }
    let vectors = effect_vectors(o, degree, raws);
    let e = vectors.len();
    for i in 0..e {
        assert!((f.effect_means[i] - dot(&vectors[i], &o.mean)).abs() < 1e-9);
        for j in 0..e {
            let expected = quad(&o.cov, &vectors[i], &vectors[j]);
            assert!((f.effect_covariance[i * e + j] - expected).abs() < 1e-9, "effect cov {i},{j}");
        }
    }
    let t = vectors.last().unwrap();
    assert!((f.target_effect_mean - dot(t, &o.mean)).abs() < 1e-9);
    assert!((f.target_effect_variance - quad(&o.cov, t, t)).abs() < 1e-9);
}

fn sample_mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

fn sample_cov(a: &[f64], b: &[f64]) -> f64 {
    let (ma, mb) = (sample_mean(a), sample_mean(b));
    a.iter().zip(b).map(|(x, y)| (x - ma) * (y - mb)).sum::<f64>() / a.len() as f64
}

fn assert_sample_moments(draws: &JointDraws, names: &[String], mean: &[f64], cov: &[Vec<f64>]) {
    let n = draws.n_draws as f64;
    let cols: Vec<Vec<f64>> =
        names.iter().map(|name| draws.coordinate(draws.column(name).unwrap())).collect();
    for i in 0..names.len() {
        let tol = 5.0 * (cov[i][i] / n).sqrt();
        assert!((sample_mean(&cols[i]) - mean[i]).abs() < tol, "sample mean {}", names[i]);
        for j in i..names.len() {
            let tol = 6.0 * ((cov[i][j] * cov[i][j] + cov[i][i] * cov[j][j]) / n).sqrt() + 1e-12;
            let got = sample_cov(&cols[i], &cols[j]);
            assert!((got - cov[i][j]).abs() < tol, "sample cov {},{}", names[i], names[j]);
        }
    }
}

fn check_draws_against_oracle(f: &LearnedJointFit, o: &Oracle, degree: usize, raws: &[&Raw]) {
    assert_sample_moments(&f.draws, &f.parameter_names, &o.mean, &o.cov);
    let vectors = effect_vectors(o, degree, raws);
    let means: Vec<f64> = vectors.iter().map(|c| dot(c, &o.mean)).collect();
    let cov: Vec<Vec<f64>> =
        vectors.iter().map(|a| vectors.iter().map(|b| quad(&o.cov, a, b)).collect()).collect();
    assert_sample_moments(&f.draws, &f.effect_names, &means, &cov);
}

fn iso_spec<'a>(degree: usize, raws: &'a [&'a Raw], vc: bool, shared: bool) -> Spec<'a> {
    let (q, _) = dims(
        degree,
        if vc { VaryingBlock::InterceptAndCovariates } else { VaryingBlock::Intercept },
    );
    Spec {
        degree,
        vc,
        shared,
        theta_mean: vec![0.0; q],
        theta_cov: iso(q, 4.0),
        gamma_var: 4.0,
        raws,
    }
}

fn wiggly(n: usize, offset: usize, b0: f64, noise: f64) -> Raw {
    raw(n, offset, b0, 2.0, (1.5, 1.2), (noise, 0.3))
}

// ---------------------------------------------------------------- positive oracle

#[test]
fn a2_learned_quadratic_basis_single_source_matches_closed_form() {
    let r1 = wiggly(60, 0, 1.0, 0.09);
    let sources = [to_source("s1", &r1, "snap-s1", "s1")];
    let f = fit(&model(), &sources, &options()).unwrap();
    let raws = [&r1];
    let o = oracle(&iso_spec(2, &raws, false, false));
    assert_matches_oracle(&f, &o, 2, &raws);
    check_draws_against_oracle(&f, &o, 2, &raws);
    // Aligned joint draws: the source-target effect covariance is nonzero (shared theta).
    let e = f.effect_names.len();
    let cross = f.effect_covariance[1];
    assert!(cross.abs() > 1e-6);
    let s = f.draws.coordinate(f.draws.column("effect.source.s1").unwrap());
    let t = f.draws.coordinate(f.draws.column("effect.target").unwrap());
    let tol = 6.0
        * ((cross * cross + f.effect_covariance[0] * f.effect_covariance[e * e - 1])
            / DRAWS as f64)
            .sqrt();
    assert!((sample_cov(&s, &t) - cross).abs() < tol);
    assert_eq!(f.draws.width(), f.parameter_names.len() + e);
    assert_eq!(
        f.parameter_names,
        [
            "theta.treatment",
            "theta.treatment:x7",
            "theta.treatment:x7^2",
            "theta.x7",
            "theta.x7^2",
            "gamma.s1.intercept",
        ]
    );
}

#[test]
fn a2_learned_two_sources_independent_and_shared_blocks_degrees_two_and_three() {
    let r1 = wiggly(50, 0, 1.0, 0.09);
    let r2 = wiggly(45, 200, -0.5, 0.16);
    let sources = [to_source("s1", &r1, "snap-s1", "s1"), to_source("s2", &r2, "snap-s2", "s2")];
    let raws = [&r1, &r2];
    for degree in [2, 3] {
        for shared in [false, true] {
            let sharing = if shared {
                SourceSharing::SharedVaryingBlock
            } else {
                SourceSharing::IndependentVaryingBlocks
            };
            let m = learned_model(degree, VaryingBlock::Intercept, sharing, 0.0, 4.0);
            let f = fit(&m, &sources, &options()).unwrap();
            let o = oracle(&iso_spec(degree, &raws, false, shared));
            assert_matches_oracle(&f, &o, degree, &raws);
            check_draws_against_oracle(&f, &o, degree, &raws);
            let blocks = if shared { 1 } else { 2 };
            assert_eq!(f.parameter_names.len(), 1 + 2 * degree + blocks);
        }
    }
}

#[test]
fn a2_learned_varying_covariate_block_and_dense_prior_match_closed_form() {
    let r1 = wiggly(60, 0, 1.0, 0.09);
    let r2 = wiggly(55, 300, 2.0, 0.09);
    let sources = [to_source("s1", &r1, "snap-s1", "s1"), to_source("s2", &r2, "snap-s2", "s2")];
    let raws = [&r1, &r2];
    // Dense invariant prior with a nonzero mean: correlated coefficients.
    let degree = 2;
    let (q, _) = dims(degree, VaryingBlock::InterceptAndCovariates);
    let mut m = learned_model(
        degree,
        VaryingBlock::InterceptAndCovariates,
        SourceSharing::IndependentVaryingBlocks,
        0.0,
        4.0,
    );
    let mut cov = iso(q, 2.0);
    for i in 0..q {
        for j in 0..q {
            if i != j {
                cov[i][j] = 0.3;
            }
        }
    }
    let theta_mean: Vec<f64> = (0..q).map(|i| 0.5 - 0.25 * i as f64).collect();
    m.priors.invariant = GaussianPrior {
        mean: theta_mean.clone(),
        covariance: cov.iter().flatten().copied().collect(),
        provenance: PriorProvenance::Declared,
    };
    let f = fit(&m, &sources, &options()).unwrap();
    let mut spec = iso_spec(degree, &raws, true, false);
    spec.theta_mean = theta_mean;
    spec.theta_cov = cov;
    let o = oracle(&spec);
    assert_matches_oracle(&f, &o, degree, &raws);
    check_draws_against_oracle(&f, &o, degree, &raws);
}

#[test]
fn a2_learned_linear_basis_reproduces_joint_bayesian_moments() {
    // Labelled regression cross-check between two Antecedent paths (not independent truth):
    // with a degree-one basis the learned provider and the joint closed-form engine are the
    // same model, so their exact moments agree.
    let r1 = wiggly(50, 0, 1.0, 0.09);
    let r2 = wiggly(45, 200, -0.5, 0.16);
    let sources = [to_source("s1", &r1, "snap-s1", "s1"), to_source("s2", &r2, "snap-s2", "s2")];
    let ctx = ExecutionContext::for_tests(1);
    for varying in [VaryingBlock::Intercept, VaryingBlock::InterceptAndCovariates] {
        for sharing in [SourceSharing::IndependentVaryingBlocks, SourceSharing::SharedVaryingBlock]
        {
            let learned_m = learned_model(1, varying, sharing, 0.0, 4.0);
            let joint_m = JointTransportModel {
                graph: learned_m.graph,
                features: learned_m.features.clone(),
                varying,
                sharing,
                dependence: learned_m.dependence,
                priors: learned_m.priors.clone(),
                max_unsupported_mass: 0.0,
                conflict_z_threshold: 3.0,
            };
            let opts = JointTransportOptions { draws: 100, seed: 5 };
            let learned = fit(&learned_m, &sources, &opts).unwrap();
            let joint = fit_joint_bayesian_transport(
                &identified(),
                &joint_m,
                &sources,
                Some(&target()),
                &opts,
                &ctx,
            )
            .unwrap();
            assert_eq!(learned.parameter_names, joint.parameter_names);
            assert_eq!(learned.effect_names, joint.effect_names);
            for (a, b) in learned.posterior_mean.iter().zip(&joint.posterior_mean) {
                assert!((a - b).abs() < 1e-9);
            }
            for (a, b) in learned.posterior_covariance.iter().zip(&joint.posterior_covariance) {
                assert!((a - b).abs() < 1e-9);
            }
            for (a, b) in learned.effect_means.iter().zip(&joint.effect_means) {
                assert!((a - b).abs() < 1e-9);
            }
            for (a, b) in learned.effect_covariance.iter().zip(&joint.effect_covariance) {
                assert!((a - b).abs() < 1e-9);
            }
            assert!((learned.target_effect_mean - joint.target_effect_mean).abs() < 1e-9);
            let (ld, jd) = (&learned.diagnostics.disagreement, &joint.diagnostics.disagreement);
            assert_eq!(ld.per_source.len(), jd.per_source.len());
            for (a, b) in ld.per_source.iter().zip(&jd.per_source) {
                assert!((a.mean - b.mean).abs() < 1e-9 && (a.variance - b.variance).abs() < 1e-9);
            }
            assert_eq!(learned.identification, joint.identification);
        }
    }
}

#[test]
fn a2_learned_nonlinear_truth_is_recovered_on_an_exact_design() {
    // Noise-free data whose mean function lies exactly in the quadratic basis span, a very
    // weak prior and a tiny declared noise: the posterior concentrates on the exact truth.
    let exact = raw(60, 0, 1.0, 2.0, (1.5, 1.2), (1e-4, 0.0));
    let sources = [to_source("s1", &exact, "snap-s1", "s1")];
    let mut m = learned_model(
        2,
        VaryingBlock::Intercept,
        SourceSharing::IndependentVaryingBlocks,
        0.0,
        1e6,
    );
    m.priors.varying = GaussianPrior::isotropic(1, 0.0, 1e6, PriorProvenance::Declared);
    let f = fit(&m, &sources, &small()).unwrap();
    let get = |name: &str| {
        let i = f.parameter_names.iter().position(|n| n == name).expect(name);
        f.posterior_mean[i]
    };
    assert!((get("theta.treatment") - 2.0).abs() < 1e-5);
    assert!((get("theta.treatment:x7") - 1.5).abs() < 1e-5);
    assert!((get("theta.treatment:x7^2") - 1.2).abs() < 1e-5);
    assert!((get("theta.x7") - 0.5).abs() < 1e-5);
    assert!((get("theta.x7^2") - 0.4).abs() < 1e-5);
    assert!((get("gamma.s1.intercept") - 1.0).abs() < 1e-5);
    let tx = target_x();
    let truth: f64 = tx.iter().map(|x| 2.0 + 1.5 * x + 1.2 * x * x).sum::<f64>() / tx.len() as f64;
    assert!((f.target_effect_mean - truth).abs() < 1e-5, "{} vs {truth}", f.target_effect_mean);
    // The declared noise variance is 1e-4; 60 rows pin the effect far tighter than one
    // observation would, but not to zero (the likelihood is not degenerate).
    assert!(
        f.target_effect_variance > 0.0 && f.target_effect_variance < 1e-4,
        "variance {}",
        f.target_effect_variance
    );
    // The same data under the closed form agrees with the fit, so the recovery is the
    // oracle's posterior, not a coincidence of the engine.
    let raws = [&exact];
    let mut spec = iso_spec(2, &raws, false, false);
    spec.theta_cov = iso(5, 1e6);
    spec.gamma_var = 1e6;
    assert_matches_oracle(&f, &oracle(&spec), 2, &raws);
}

// ---------------------------------------------------------------- identification and guards

#[test]
fn a2_learned_prior_strength_never_changes_identification_status() {
    let r1 = wiggly(30, 0, 1.0, 0.09);
    let sources = [to_source("s1", &r1, "snap-s1", "s1")];
    let mut records = Vec::new();
    for var in [1e-8, 1.0, 1e8] {
        let m = learned_model(
            2,
            VaryingBlock::Intercept,
            SourceSharing::IndependentVaryingBlocks,
            3.0,
            var,
        );
        records.push(fit(&m, &sources, &small()).unwrap().identification);
    }
    assert!(records.windows(2).all(|w| w[0] == w[1]));
    assert_eq!(records[0].status, "identified");
    assert_eq!(records[0].formula, "standardize");
    assert_eq!(records[0].rule, "standardization");
    // An infinitely confident prior cannot rescue an uncertified derivation.
    let tight = learned_model(
        2,
        VaryingBlock::Intercept,
        SourceSharing::IndependentVaryingBlocks,
        2.0,
        1e-12,
    );
    let ctx = ExecutionContext::for_tests(1);
    let not_certified = TransportIdentification::NotCertified(NotCertifiedCertificate {
        reason: Arc::from("selection_on_outcome"),
        witness: Arc::from([]),
        message: Arc::from("not identified"),
    });
    let refusal = fit_learned_joint_transport(
        &not_certified,
        &tight,
        &sources,
        Some(&target()),
        &small(),
        &ctx,
    )
    .unwrap_err();
    assert_eq!(refusal.code, "transport_not_certified");
    assert_eq!(refusal.detail, DETAIL_IDENTIFICATION);
    let missing = TransportIdentification::MissingEvidence(MissingEvidenceCertificate {
        reason: Arc::from("absent"),
        missing: Arc::from([]),
        message: Arc::from("none"),
    });
    let refusal =
        fit_learned_joint_transport(&missing, &model(), &sources, Some(&target()), &small(), &ctx)
            .unwrap_err();
    assert_eq!(refusal.code, "transport_missing_evidence");
}

#[test]
fn a2_learned_prior_bank_and_likelihood_double_use_refuses() {
    let r1 = wiggly(30, 0, 1.0, 0.09);
    let sources = [to_source("s1", &r1, "snap-s1", "s1")];
    let mut m = model();
    m.priors.invariant.provenance = PriorProvenance::Bank {
        bank_id: "bank-1".into(),
        consumed: vec![DataIdentity { snapshot_digest: "snap-s1".into(), datum_ids: vec![] }],
    };
    let refusal = refusal_of(&m, &sources, Some(&target()));
    assert_eq!(refusal.code, "invalid_argument");
    assert_eq!(refusal.detail, DETAIL_PRIOR_DATA_OVERLAP);
    assert!(refusal.failure.overlapping_ids.contains(&"snap-s1".to_owned()));
    // One reused datum under a different snapshot digest is still double use.
    let mut m = model();
    m.priors.varying.provenance = PriorProvenance::Bank {
        bank_id: "bank-2".into(),
        consumed: vec![DataIdentity {
            snapshot_digest: "other-snapshot".into(),
            datum_ids: vec!["s1-3".into(), "elsewhere-1".into()],
        }],
    };
    let refusal = refusal_of(&m, &sources, Some(&target()));
    assert_eq!(refusal.detail, DETAIL_PRIOR_DATA_OVERLAP);
    assert_eq!(refusal.failure.overlapping_ids, vec!["s1-3".to_owned()]);
    // A bank built from disjoint data is accepted.
    let mut m = model();
    m.priors.invariant.provenance = PriorProvenance::Bank {
        bank_id: "bank-3".into(),
        consumed: vec![DataIdentity {
            snapshot_digest: "earlier-snapshot".into(),
            datum_ids: vec!["old-1".into()],
        }],
    };
    assert!(fit(&m, &sources, &small()).is_ok());
}

#[test]
fn a2_learned_weak_overlap_refuses_with_retained_diagnostics() {
    let r1 = wiggly(40, 0, 1.0, 0.09);
    let sources = [to_source("s1", &r1, "snap-s1", "s1")];
    let mut shifted = target();
    for (j, value) in shifted.covariates[0].iter_mut().enumerate() {
        if j % 2 == 0 {
            *value += 50.0;
        }
    }
    let refusal = refusal_of(&model(), &sources, Some(&shifted));
    assert_eq!(refusal.code, "transport_support_failure");
    assert_eq!(refusal.detail, DETAIL_WEAK_OVERLAP);
    assert!(refusal.failure.unsupported_mass.is_some_and(|mass| (mass - 0.5).abs() < 1e-12));
    assert!(refusal.failure.tolerance.is_some());
    assert!(EstimationError::from(refusal).to_string().contains(DETAIL_WEAK_OVERLAP));
    // A declared tolerance above the unsupported mass admits the same target.
    let mut lenient = model();
    lenient.max_unsupported_mass = 0.6;
    let ctx = ExecutionContext::for_tests(1);
    let ok = fit_learned_joint_transport(
        &identified(),
        &lenient,
        &sources,
        Some(&shifted),
        &small(),
        &ctx,
    )
    .unwrap();
    assert!((ok.diagnostics.overlap.unsupported_mass - 0.5).abs() < 1e-12);
}

#[test]
fn a2_learned_rank_deficient_basis_refuses_with_failure_diagnostics() {
    // Every source covariate equals 0.5: x and x^2 are constant, so the basis columns are
    // collinear with the intercept and the treatment column.
    let mut r1 = wiggly(30, 0, 1.0, 0.09);
    r1.x = vec![0.5; 30];
    let sources = [to_source("s1", &r1, "snap-s1", "s1")];
    let flat_target = TargetData { covariates: vec![vec![0.5; 40]], ..target() };
    let refusal = refusal_of(&model(), &sources, Some(&flat_target));
    assert_eq!(refusal.code, "design_rank_deficient");
    assert_eq!(refusal.detail, DETAIL_RANK_DEFICIENT);
    assert_eq!(refusal.failure.stage, "fit");
    assert!(refusal.failure.min_pivot_ratio.is_some_and(|ratio| ratio <= 1e-10));
    assert!(refusal.failure.failing_column.is_some());
    assert!(EstimationError::from(refusal).to_string().contains(DETAIL_RANK_DEFICIENT));
    // Every unit treated: the treatment column is collinear with the intercept.
    let mut r2 = wiggly(30, 0, 1.0, 0.09);
    r2.a = vec![true; 30];
    let sources = [to_source("s1", &r2, "snap-s1", "s1")];
    assert_eq!(refusal_of(&model(), &sources, Some(&target())).detail, DETAIL_RANK_DEFICIENT);
    // A healthy design under the same model is accepted.
    let healthy = wiggly(30, 0, 1.0, 0.09);
    assert!(fit(&model(), &[to_source("s1", &healthy, "snap-s1", "s1")], &small()).is_ok());
}

#[test]
fn a2_learned_conflicting_sources_are_reported_not_silently_pooled() {
    let r1 = raw(60, 0, 1.0, 3.0, (0.0, 0.0), (0.04, 0.3));
    let r2 = raw(60, 500, 1.0, -3.0, (0.0, 0.0), (0.04, 0.3));
    let conflicting =
        [to_source("s1", &r1, "snap-s1", "s1"), to_source("s2", &r2, "snap-s2", "s2")];
    let f = fit(&model(), &conflicting, &JointTransportOptions { draws: 200, seed: 3 }).unwrap();
    let d = &f.diagnostics.disagreement;
    assert!(d.flagged);
    assert_eq!(d.per_source.len(), 2);
    assert_eq!(d.pairs.len(), 1);
    assert!(d.max_abs_z > 10.0, "z {}", d.max_abs_z);
    assert!(d.per_source[0].mean > 2.0 && d.per_source[1].mean < -2.0);
    // The pooled posterior is still reported (same closed form), next to the warning.
    let raws = [&r1, &r2];
    assert_matches_oracle(&f, &oracle(&iso_spec(2, &raws, false, false)), 2, &raws);
    // Consistent sources are not flagged.
    let r3 = raw(60, 900, 1.0, 3.0, (0.0, 0.0), (0.04, 0.3));
    let agreeing = [to_source("s1", &r1, "snap-s1", "s1"), to_source("s3", &r3, "snap-s3", "s3")];
    let g = fit(&model(), &agreeing, &JointTransportOptions { draws: 200, seed: 3 }).unwrap();
    assert!(!g.diagnostics.disagreement.flagged);
    assert!(g.diagnostics.disagreement.max_abs_z < 3.0);
    // A single source has no disagreement to report.
    let one = fit(&model(), &conflicting[..1], &small()).unwrap();
    assert!(one.diagnostics.disagreement.pairs.is_empty());
}

#[test]
fn a2_learned_draws_are_seeded_deterministic_and_bounded() {
    let r1 = wiggly(30, 0, 1.0, 0.09);
    let sources = [to_source("s1", &r1, "snap-s1", "s1")];
    let few = JointTransportOptions { draws: 500, seed: 9 };
    let a = fit(&model(), &sources, &few).unwrap();
    let b = fit(&model(), &sources, &few).unwrap();
    let c = fit(&model(), &sources, &JointTransportOptions { draws: 500, seed: 10 }).unwrap();
    assert_eq!(a, b);
    assert_ne!(a.draws.values, c.draws.values);
    assert!(a.draws.rng_id.contains("seed=9"));
    assert_eq!(a.posterior_mean, c.posterior_mean);
    assert!((a.diagnostics.effective_sample_size - 500.0).abs() < 1e-12);
    assert_eq!(a.diagnostics.draw_count, 500);
    assert_eq!(a.diagnostics.sampler, LEARNED_JOINT_SAMPLER);
    assert!(a.diagnostics.r_hat.is_none());
    assert_eq!(a.diagnostics.basis_rank, a.parameter_names.len());
    assert!(a.diagnostics.precision_condition.is_finite());
    assert_eq!(a.diagnostics.calibration, JointTransportCalibration::Unmeasured);
    let top = fit(&model(), &sources, &JointTransportOptions { draws: 100_000, seed: 1 }).unwrap();
    assert_eq!(top.draws.n_draws, 100_000);
    for bad in [0, 100_001] {
        let refusal =
            fit(&model(), &sources, &JointTransportOptions { draws: bad, seed: 1 }).unwrap_err();
        assert_eq!(refusal.code, "invalid_argument");
        assert_eq!(refusal.detail, DETAIL_TOO_MANY_DRAWS);
    }
    let ctx = ExecutionContext::for_tests(1);
    ctx.cancellation.cancel();
    let refusal = fit_learned_joint_transport(
        &identified(),
        &model(),
        &sources,
        Some(&target()),
        &small(),
        &ctx,
    )
    .unwrap_err();
    assert_eq!(refusal.code, "transport_budget_cancel");
    assert_eq!(refusal.detail, DETAIL_CANCELLED);
}

#[test]
fn a2_learned_provider_record_and_effect_draws_name_the_learn_row() {
    let r1 = wiggly(30, 0, 1.0, 0.09);
    let r2 = wiggly(30, 100, 1.0, 0.09);
    let sources = [to_source("s1", &r1, "snap-s1", "s1"), to_source("s2", &r2, "snap-s2", "s2")];
    let f = fit(&model(), &sources, &small()).unwrap();
    assert_eq!(f.provider.provider, LEARNED_JOINT_PROVIDER);
    assert_eq!(f.provider.provider, "antecedent-learn");
    assert_eq!(f.provider.graph, "fixed_dag");
    assert_eq!(f.provider.query, LEARNED_JOINT_QUERY);
    assert_eq!(f.provider.basis_id, "polynomial_degree_2");
    assert_eq!(f.provider.basis_terms, ["x7", "x7^2"]);
    assert_eq!(f.provider.learn_model_id, "bayesian_basis.known_variance_gaussian");
    assert!(f.provider.learn_implementation.contains("conjugate_gaussian"));
    assert!(f.model_identity.starts_with("learned_joint_transport_v1|provider=antecedent-learn"));
    let (names, values) = f.effect_draws();
    assert_eq!(names, ["effect.source.s1", "effect.source.s2", "effect.target"]);
    assert_eq!(values.len(), 50 * 3);
}

#[test]
fn a2_learned_declaration_and_input_refusals_carry_typed_details() {
    let r1 = wiggly(30, 0, 1.0, 0.09);
    let sources = [to_source("s1", &r1, "snap-s1", "s1")];
    for graph in [TransportGraphClass::Admg, TransportGraphClass::GraphPosterior] {
        let mut m = model();
        m.graph = graph;
        let refusal = refusal_of(&m, &sources, Some(&target()));
        assert_eq!(refusal.code, "route_not_supported");
        assert_eq!(refusal.detail, DETAIL_UNSUPPORTED_GRAPH);
    }
    let frozen = route_frozen_refusal();
    assert_eq!(frozen.code, "cell_not_licensed");
    assert_eq!(frozen.detail, DETAIL_ROUTE_FROZEN);
    let mut m = model();
    m.features = vec![FEATURE + 1];
    assert_eq!(refusal_of(&m, &sources, Some(&target())).detail, DETAIL_FEATURE_MISMATCH);
    for degree in [0, 7] {
        let mut m = model();
        m.basis_degree = degree;
        assert_eq!(refusal_of(&m, &sources, Some(&target())).detail, DETAIL_INVALID_BASIS);
    }
    for dependence in [SourceDependence::OverlappingUnits, SourceDependence::Unknown] {
        let mut m = model();
        m.dependence = dependence;
        let refusal = refusal_of(&m, &sources, Some(&target()));
        assert_eq!(refusal.code, "sampling_dependence_unknown");
        assert_eq!(refusal.detail, DETAIL_SOURCE_DEPENDENCE);
    }
    let r2 = wiggly(30, 100, 1.0, 0.09);
    let both = [to_source("s1", &r1, "snap-s1", "dup"), to_source("s2", &r2, "snap-s2", "dup")];
    let refusal = refusal_of(&model(), &both, Some(&target()));
    assert_eq!(refusal.detail, DETAIL_SOURCE_DEPENDENCE);
    assert!(!refusal.failure.overlapping_ids.is_empty());
    // Priors of the wrong size, asymmetric or not positive definite.
    let mut m = model();
    m.priors.invariant = GaussianPrior::isotropic(2, 0.0, 1.0, PriorProvenance::Declared);
    let refusal = refusal_of(&m, &sources, Some(&target()));
    assert_eq!(refusal.code, "prior_dimension_mismatch");
    assert_eq!(refusal.detail, DETAIL_PRIOR_DIMENSION);
    let mut m = model();
    m.priors.invariant.covariance[0] = -1.0;
    assert_eq!(refusal_of(&m, &sources, Some(&target())).detail, DETAIL_INVALID_PRIOR);
    let mut m = model();
    let dim = m.priors.invariant.mean.len();
    m.priors.invariant.covariance[1] = 0.5;
    m.priors.invariant.covariance[dim] = 0.1;
    assert_eq!(refusal_of(&m, &sources, Some(&target())).detail, DETAIL_INVALID_PRIOR);
    // Missing law, empty target, missing source, malformed source.
    let refusal = refusal_of(&model(), &sources, None);
    assert_eq!(refusal.code, "joint_law_required");
    assert_eq!(refusal.detail, DETAIL_MISSING_LAW);
    let empty = TargetData { rows: 0, covariates: vec![vec![]], ..target() };
    assert_eq!(refusal_of(&model(), &sources, Some(&empty)).detail, DETAIL_MISSING_LAW);
    let refusal = refusal_of(&model(), &[], Some(&target()));
    assert_eq!(refusal.code, "transport_missing_evidence");
    assert_eq!(refusal.detail, DETAIL_MISSING_SOURCE);
    let mut bad = sources[0].clone();
    bad.noise_variance = 0.0;
    assert_eq!(refusal_of(&model(), &[bad], Some(&target())).detail, DETAIL_INVALID_INPUT);
    // Every detail is a plain namespaced literal.
    for detail in [
        DETAIL_CANCELLED,
        DETAIL_FEATURE_MISMATCH,
        DETAIL_FIT_FAILED,
        DETAIL_IDENTIFICATION,
        DETAIL_INVALID_BASIS,
        DETAIL_INVALID_INPUT,
        DETAIL_INVALID_PRIOR,
        DETAIL_MISSING_LAW,
        DETAIL_MISSING_SOURCE,
        DETAIL_PRIOR_DATA_OVERLAP,
        DETAIL_PRIOR_DIMENSION,
        DETAIL_RANK_DEFICIENT,
        DETAIL_ROUTE_FROZEN,
        DETAIL_SOURCE_DEPENDENCE,
        DETAIL_TOO_MANY_DRAWS,
        DETAIL_TOO_MANY_PARAMETERS,
        DETAIL_UNSUPPORTED_GRAPH,
        DETAIL_WEAK_OVERLAP,
    ] {
        assert!(detail.starts_with("learned_joint_transport."), "{detail}");
    }
}

#[test]
fn a2_learned_too_many_parameters_refuses() {
    let p = 130;
    let over: Vec<VariableId> = (0..p).map(VariableId::from_raw).collect();
    let id = TransportIdentification::Transportable {
        formula: TransportFormula::Standardize {
            over: Arc::from(over),
            source_response: factor(),
            target_law: factor(),
        },
        certificate: certificate(),
    };
    let n = 12;
    let wide_source = SourceData {
        id: "wide".into(),
        identity: DataIdentity { snapshot_digest: "w".into(), datum_ids: vec![] },
        treatment: (0..n).map(|i| i % 2 == 0).collect(),
        outcome: vec![0.0; n],
        covariates: vec![vec![0.0; n]; p as usize],
        noise_variance: 1.0,
    };
    let wide_target = TargetData {
        identity: DataIdentity { snapshot_digest: "wt".into(), datum_ids: vec![] },
        rows: 3,
        covariates: vec![vec![0.0; 3]; p as usize],
    };
    let mut m = model();
    m.features = (0..p).collect();
    let ctx = ExecutionContext::for_tests(1);
    let refusal =
        fit_learned_joint_transport(&id, &m, &[wide_source], Some(&wide_target), &options(), &ctx)
            .unwrap_err();
    assert_eq!(refusal.code, "invalid_argument");
    assert_eq!(refusal.detail, DETAIL_TOO_MANY_PARAMETERS);
}
