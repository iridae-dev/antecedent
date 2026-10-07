//! 2.3A X4: joint Bayesian source-target transport against an independent exact oracle.
//!
//! The oracle below is written out in closed-form matrix algebra (Gauss-Jordan inverse of
//! the posterior precision) and never calls the code under test. Calibration is
//! unmeasured in this wave; nothing here asserts coverage.
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
    DETAIL_CANCELLED, DETAIL_FEATURE_MISMATCH, DETAIL_INVALID_PRIOR, DETAIL_MISSING_LAW,
    DETAIL_PRIOR_DATA_OVERLAP, DETAIL_PRIOR_DIMENSION, DETAIL_ROUTE_FROZEN, DETAIL_SINGULAR_FIT,
    DETAIL_SOURCE_DEPENDENCE, DETAIL_TOO_MANY_DRAWS, DETAIL_TOO_MANY_PARAMETERS,
    DETAIL_UNSUPPORTED_GRAPH, DETAIL_WEAK_OVERLAP, DataIdentity, GaussianPrior, JointDraws,
    JointPriors, JointTransportCalibration, JointTransportFit, JointTransportModel,
    JointTransportOptions, JointTransportRefusal, PriorProvenance, SourceData, SourceDependence,
    SourceSharing, TargetData, TransportGraphClass, VaryingBlock, fit_joint_bayesian_transport,
    route_frozen_refusal,
};
use antecedent_identify::{
    MissingEvidenceCertificate, NotCertifiedCertificate, PopulationFactor, TransportCertificate,
    TransportFormula, TransportIdentification,
};
use std::sync::Arc;

const DRAWS: usize = 60_000;
const FEATURE: u32 = 7;

// ---------------------------------------------------------------- fixtures

struct Raw {
    x: Vec<f64>,
    a: Vec<bool>,
    y: Vec<f64>,
    noise: f64,
}

fn raw(n: usize, offset: usize, intercept: f64, effect: f64, slope: f64, noise: f64) -> Raw {
    let mut r = Raw { x: vec![], a: vec![], y: vec![], noise };
    for i in 0..n {
        let k = (i + offset) as f64;
        let x = 1.5 * (0.37 * k).sin() + 0.2;
        let a = i % 2 == 0;
        let treat = if a { 1.0 } else { 0.0 };
        let y = intercept + 0.5 * x + treat * (effect + slope * x) + 0.3 * (1.3 * k).cos();
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

fn model_with(
    varying: VaryingBlock,
    sharing: SourceSharing,
    theta_mean: f64,
    theta_var: f64,
) -> JointTransportModel {
    let (q, r) = match varying {
        VaryingBlock::Intercept => (3, 1),
        VaryingBlock::InterceptAndCovariates => (2, 2),
    };
    JointTransportModel {
        graph: TransportGraphClass::FixedDag,
        features: vec![FEATURE],
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

fn model() -> JointTransportModel {
    model_with(VaryingBlock::Intercept, SourceSharing::IndependentVaryingBlocks, 0.0, 4.0)
}

fn options() -> JointTransportOptions {
    JointTransportOptions { draws: DRAWS, seed: 20_230_903 }
}

// The public refusal type is deliberately rich; tests exercise it unboxed.
#[allow(clippy::result_large_err)]
fn fit(
    model: &JointTransportModel,
    sources: &[SourceData],
    options: &JointTransportOptions,
) -> Result<JointTransportFit, JointTransportRefusal> {
    let ctx = ExecutionContext::for_tests(1);
    fit_joint_bayesian_transport(&identified(), model, sources, Some(&target()), options, &ctx)
}

fn refusal_of(
    model: &JointTransportModel,
    sources: &[SourceData],
    target: Option<&TargetData>,
) -> JointTransportRefusal {
    let ctx = ExecutionContext::for_tests(1);
    fit_joint_bayesian_transport(&identified(), model, sources, target, &options(), &ctx)
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

struct Oracle {
    mean: Vec<f64>,
    cov: Vec<Vec<f64>>,
    dim: usize,
}

#[derive(Clone, Copy)]
struct Spec<'a> {
    vc: bool,
    shared: bool,
    theta_mean: f64,
    theta_var: f64,
    gamma_var: f64,
    raws: &'a [&'a Raw],
}

fn oracle(spec: Spec<'_>) -> Oracle {
    let q = if spec.vc { 2 } else { 3 };
    let r = if spec.vc { 2 } else { 1 };
    let blocks = if spec.shared { 1 } else { spec.raws.len() };
    let dim = q + blocks * r;
    let mut precision = vec![vec![0.0; dim]; dim];
    let mut rhs = vec![0.0; dim];
    for i in 0..dim {
        if i < q {
            precision[i][i] = 1.0 / spec.theta_var;
            rhs[i] = spec.theta_mean / spec.theta_var;
        } else {
            precision[i][i] = 1.0 / spec.gamma_var;
        }
    }
    for (s, rw) in spec.raws.iter().enumerate() {
        let off = q + if spec.shared { 0 } else { s } * r;
        for i in 0..rw.y.len() {
            let a = if rw.a[i] { 1.0 } else { 0.0 };
            let x = rw.x[i];
            let mut z = vec![0.0; dim];
            z[0] = a;
            z[1] = a * x;
            if !spec.vc {
                z[2] = x;
            }
            z[off] = 1.0;
            if spec.vc {
                z[off + 1] = x;
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

fn effect_vec(dim: usize, xbar: f64) -> Vec<f64> {
    let mut c = vec![0.0; dim];
    c[0] = 1.0;
    c[1] = xbar;
    c
}

fn xbar(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
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
fn effect_vectors(o: &Oracle, raws: &[&Raw]) -> Vec<Vec<f64>> {
    let mut v: Vec<Vec<f64>> = raws.iter().map(|r| effect_vec(o.dim, xbar(&r.x))).collect();
    v.push(effect_vec(o.dim, xbar(&target_x())));
    v
}

// ---------------------------------------------------------------- comparison helpers

fn assert_engine_matches_oracle(f: &JointTransportFit, o: &Oracle, raws: &[&Raw]) {
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
    let vectors = effect_vectors(o, raws);
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

/// Sample moments of named coordinates against analytic ones, tolerances scaled to N.
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

fn check_draws_against_oracle(f: &JointTransportFit, o: &Oracle, raws: &[&Raw]) {
    assert_sample_moments(&f.draws, &f.parameter_names, &o.mean, &o.cov);
    let vectors = effect_vectors(o, raws);
    let means: Vec<f64> = vectors.iter().map(|c| dot(c, &o.mean)).collect();
    let cov: Vec<Vec<f64>> =
        vectors.iter().map(|a| vectors.iter().map(|b| quad(&o.cov, a, b)).collect()).collect();
    assert_sample_moments(&f.draws, &f.effect_names, &means, &cov);
}

fn spec<'a>(raws: &'a [&'a Raw], vc: bool, shared: bool, tm: f64, tv: f64) -> Spec<'a> {
    Spec { vc, shared, theta_mean: tm, theta_var: tv, gamma_var: 4.0, raws }
}

// ---------------------------------------------------------------- positive oracle

#[test]
fn x4_joint_posterior_oracle_single_source_matches_closed_form() {
    let r1 = raw(60, 0, 1.0, 2.0, 1.5, 0.09);
    let sources = [to_source("s1", &r1, "snap-s1", "s1")];
    let f = fit(&model(), &sources, &options()).unwrap();
    let raws = [&r1];
    let o = oracle(spec(&raws, false, false, 0.0, 4.0));
    assert_engine_matches_oracle(&f, &o, &raws);
    check_draws_against_oracle(&f, &o, &raws);
    // Aligned joint draws: the source-target effect covariance is the analytic one, which
    // is nonzero because both effects are functions of the shared theta.
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
    assert_eq!(f.draws.n_draws, DRAWS);
    assert_eq!(f.draws.width(), f.parameter_names.len() + e);
}

#[test]
fn x4_joint_posterior_oracle_two_sources_independent_and_shared_blocks() {
    let r1 = raw(50, 0, 1.0, 2.0, 1.5, 0.09);
    let r2 = raw(45, 200, -0.5, 2.0, 1.5, 0.16);
    let sources = [to_source("s1", &r1, "snap-s1", "s1"), to_source("s2", &r2, "snap-s2", "s2")];
    let raws = [&r1, &r2];
    for shared in [false, true] {
        let sharing = if shared {
            SourceSharing::SharedVaryingBlock
        } else {
            SourceSharing::IndependentVaryingBlocks
        };
        let m = model_with(VaryingBlock::Intercept, sharing, 0.0, 4.0);
        let f = fit(&m, &sources, &options()).unwrap();
        let o = oracle(spec(&raws, false, shared, 0.0, 4.0));
        assert_engine_matches_oracle(&f, &o, &raws);
        check_draws_against_oracle(&f, &o, &raws);
        let expected = if shared { 3 + 1 } else { 3 + 2 };
        assert_eq!(f.parameter_names.len(), expected);
    }
}

#[test]
fn x4_joint_posterior_oracle_varying_covariate_block_matches_closed_form() {
    let r1 = raw(60, 0, 1.0, 2.0, 1.5, 0.09);
    let r2 = raw(55, 300, 2.0, 2.0, 1.5, 0.09);
    let sources = [to_source("s1", &r1, "snap-s1", "s1"), to_source("s2", &r2, "snap-s2", "s2")];
    let raws = [&r1, &r2];
    let m = model_with(
        VaryingBlock::InterceptAndCovariates,
        SourceSharing::IndependentVaryingBlocks,
        0.0,
        4.0,
    );
    let f = fit(&m, &sources, &options()).unwrap();
    let o = oracle(spec(&raws, true, false, 0.0, 4.0));
    assert_engine_matches_oracle(&f, &o, &raws);
    check_draws_against_oracle(&f, &o, &raws);
}

#[test]
fn x4_joint_posterior_oracle_prior_sensitivity_moves_toward_prior_by_analytic_amount() {
    let r1 = raw(30, 0, 1.0, 2.0, 1.5, 0.25);
    let sources = [to_source("s1", &r1, "snap-s1", "s1")];
    let raws = [&r1];
    let prior_mean = 5.0;
    let c = effect_vec(3, xbar(&target_x()));
    // The prior effect mean: every theta coordinate has prior mean `prior_mean`.
    let prior_effect = c[0] * prior_mean + c[1] * prior_mean;
    let mut effects = Vec::new();
    for theta_var in [100.0, 1.0, 0.01] {
        let m = model_with(
            VaryingBlock::Intercept,
            SourceSharing::IndependentVaryingBlocks,
            prior_mean,
            theta_var,
        );
        let f = fit(&m, &sources, &options()).unwrap();
        let o = oracle(spec(&raws, false, false, prior_mean, theta_var));
        assert_engine_matches_oracle(&f, &o, &raws);
        effects.push(f.target_effect_mean);
    }
    // A tighter prior pulls the target effect strictly closer to the prior's effect mean.
    let gaps: Vec<f64> = effects.iter().map(|e| (e - prior_effect).abs()).collect();
    assert!(gaps[0] > gaps[1] && gaps[1] > gaps[2], "gaps {gaps:?}");
    // And the weakest prior is near the data-only (flat prior) closed form.
    let flat = oracle(spec(&raws, false, false, 0.0, 1e12));
    let weak = oracle(spec(&raws, false, false, prior_mean, 100.0));
    let vectors = effect_vectors(&flat, &raws);
    let t = &vectors[1];
    let moved = dot(t, &weak.mean) - dot(t, &flat.mean);
    assert!((effects[0] - dot(t, &flat.mean) - moved).abs() < 1e-9);
}

#[test]
fn x4_joint_posterior_oracle_wrong_prior_still_follows_the_closed_form() {
    let r1 = raw(40, 0, 1.0, 2.0, 0.0, 0.25);
    let sources = [to_source("s1", &r1, "snap-s1", "s1")];
    let raws = [&r1];
    // Truth: the effect is 2.0 for every x. The prior says 10 with moderate confidence.
    let m = model_with(VaryingBlock::Intercept, SourceSharing::IndependentVaryingBlocks, 10.0, 1.0);
    let f = fit(&m, &sources, &options()).unwrap();
    let o = oracle(spec(&raws, false, false, 10.0, 1.0));
    assert_engine_matches_oracle(&f, &o, &raws);
    check_draws_against_oracle(&f, &o, &raws);
    // The posterior is the closed form, which here is biased away from the truth of 2.0
    // toward the wrong prior: no hidden correction toward the data.
    let flat = oracle(spec(&raws, false, false, 0.0, 1e12));
    let t = effect_vec(flat.dim, xbar(&target_x()));
    assert!(f.target_effect_mean > dot(&t, &flat.mean) + 0.05);
}

#[test]
fn x4_joint_posterior_oracle_draws_are_seeded_and_bounded() {
    let r1 = raw(30, 0, 1.0, 2.0, 1.5, 0.09);
    let sources = [to_source("s1", &r1, "snap-s1", "s1")];
    let small = JointTransportOptions { draws: 500, seed: 9 };
    let a = fit(&model(), &sources, &small).unwrap();
    let b = fit(&model(), &sources, &small).unwrap();
    let c = fit(&model(), &sources, &JointTransportOptions { draws: 500, seed: 10 }).unwrap();
    assert_eq!(a.draws, b.draws);
    assert_ne!(a.draws.values, c.draws.values);
    assert!(a.draws.rng_id.contains("seed=9"));
    assert!((a.diagnostics.effective_sample_size - 500.0).abs() < 1e-12);
    assert_eq!(a.diagnostics.draw_count, 500);
    assert!(a.diagnostics.r_hat.is_none());
    assert_eq!(a.diagnostics.calibration, JointTransportCalibration::Unmeasured);
    assert_eq!(a.model_identity, b.model_identity);
    let top = fit(&model(), &sources, &JointTransportOptions { draws: 100_000, seed: 1 }).unwrap();
    assert_eq!(top.draws.n_draws, 100_000);
    for bad in [0, 100_001] {
        let refusal =
            fit(&model(), &sources, &JointTransportOptions { draws: bad, seed: 1 }).unwrap_err();
        assert_eq!(refusal.code, "invalid_argument");
        assert_eq!(refusal.detail, DETAIL_TOO_MANY_DRAWS);
    }
}

#[test]
fn x4_joint_posterior_oracle_cancellation_refuses_with_typed_detail() {
    let r1 = raw(30, 0, 1.0, 2.0, 1.5, 0.09);
    let sources = [to_source("s1", &r1, "snap-s1", "s1")];
    let ctx = ExecutionContext::for_tests(1);
    ctx.cancellation.cancel();
    let refusal = fit_joint_bayesian_transport(
        &identified(),
        &model(),
        &sources,
        Some(&target()),
        &options(),
        &ctx,
    )
    .unwrap_err();
    assert_eq!(refusal.code, "transport_budget_cancel");
    assert_eq!(refusal.detail, DETAIL_CANCELLED);
}

// ---------------------------------------------------------------- negatives

#[test]
fn x4_prior_data_overlap_bank_and_likelihood_refuse() {
    let r1 = raw(30, 0, 1.0, 2.0, 1.5, 0.09);
    let sources = [to_source("s1", &r1, "snap-s1", "s1")];
    // Same snapshot digest consumed by the prior bank and by the likelihood.
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
    assert!(fit(&m, &sources, &JointTransportOptions { draws: 10, seed: 1 }).is_ok());
}

#[test]
fn x4_prior_data_overlap_unsupported_graph_and_uncertified_derivation_refuse() {
    let r1 = raw(30, 0, 1.0, 2.0, 1.5, 0.09);
    let sources = [to_source("s1", &r1, "snap-s1", "s1")];
    for graph in [TransportGraphClass::Admg, TransportGraphClass::GraphPosterior] {
        let mut m = model();
        m.graph = graph;
        let refusal = refusal_of(&m, &sources, Some(&target()));
        assert_eq!(refusal.code, "route_not_supported");
        assert_eq!(refusal.detail, DETAIL_UNSUPPORTED_GRAPH);
    }
    let ctx = ExecutionContext::for_tests(1);
    let not_certified = TransportIdentification::NotCertified(NotCertifiedCertificate {
        reason: Arc::from("no_rule"),
        witness: Arc::from([]),
        message: Arc::from("none"),
    });
    let refusal = fit_joint_bayesian_transport(
        &not_certified,
        &model(),
        &sources,
        Some(&target()),
        &options(),
        &ctx,
    )
    .unwrap_err();
    assert_eq!(refusal.code, "transport_not_certified");
    let missing = TransportIdentification::MissingEvidence(MissingEvidenceCertificate {
        reason: Arc::from("absent"),
        missing: Arc::from([]),
        message: Arc::from("none"),
    });
    let refusal = fit_joint_bayesian_transport(
        &missing,
        &model(),
        &sources,
        Some(&target()),
        &options(),
        &ctx,
    )
    .unwrap_err();
    assert_eq!(refusal.code, "transport_missing_evidence");
    // The route-level refusal of the closed interval route.
    let frozen = route_frozen_refusal();
    assert_eq!(frozen.code, "cell_not_licensed");
    assert_eq!(frozen.detail, DETAIL_ROUTE_FROZEN);
    // Covariates that are not the certified standardizers are refused.
    let mut m = model();
    m.features = vec![FEATURE + 1];
    assert_eq!(refusal_of(&m, &sources, Some(&target())).detail, DETAIL_FEATURE_MISMATCH);
}

#[test]
fn x4_prior_data_overlap_singular_design_retains_failure_diagnostics() {
    // Every source unit is treated: the treatment column is collinear with the intercept.
    let mut r1 = raw(30, 0, 1.0, 2.0, 1.5, 0.09);
    r1.a = vec![true; 30];
    let sources = [to_source("s1", &r1, "snap-s1", "s1")];
    let refusal = refusal_of(&model(), &sources, Some(&target()));
    assert_eq!(refusal.code, "design_rank_deficient");
    assert_eq!(refusal.detail, DETAIL_SINGULAR_FIT);
    assert_eq!(refusal.failure.stage, "fit");
    assert!(refusal.failure.min_pivot_ratio.is_some_and(|ratio| ratio <= 1e-10));
    assert!(refusal.failure.failing_column.is_some());
    let converted = EstimationError::from(refusal);
    assert!(converted.to_string().contains(DETAIL_SINGULAR_FIT));
    // Priors that are not symmetric positive definite or of the wrong size refuse too.
    let healthy = raw(30, 0, 1.0, 2.0, 1.5, 0.09);
    let sources = [to_source("s1", &healthy, "snap-s1", "s1")];
    let mut m = model();
    m.priors.invariant.covariance[0] = -1.0;
    assert_eq!(refusal_of(&m, &sources, Some(&target())).detail, DETAIL_INVALID_PRIOR);
    let mut m = model();
    m.priors.invariant = GaussianPrior::isotropic(2, 0.0, 1.0, PriorProvenance::Declared);
    let refusal = refusal_of(&m, &sources, Some(&target()));
    assert_eq!(refusal.code, "prior_dimension_mismatch");
    assert_eq!(refusal.detail, DETAIL_PRIOR_DIMENSION);
}

#[test]
fn x4_prior_data_overlap_weak_overlap_and_missing_law_refuse() {
    let r1 = raw(40, 0, 1.0, 2.0, 1.5, 0.09);
    let sources = [to_source("s1", &r1, "snap-s1", "s1")];
    // Half of the target sits far outside the source covariate range.
    let mut shifted = target();
    for (j, value) in shifted.covariates[0].iter_mut().enumerate() {
        if j % 2 == 0 {
            *value += 50.0;
        }
    }
    let refusal = refusal_of(&model(), &sources, Some(&shifted));
    assert_eq!(refusal.code, "transport_support_failure");
    assert_eq!(refusal.detail, DETAIL_WEAK_OVERLAP);
    assert!(refusal.failure.unsupported_mass.is_some_and(|m| (m - 0.5).abs() < 1e-12));
    assert!(refusal.failure.tolerance.is_some());
    assert!(EstimationError::from(refusal).to_string().contains(DETAIL_WEAK_OVERLAP));
    // A declared tolerance above the unsupported mass admits the same target.
    let mut lenient = model();
    lenient.max_unsupported_mass = 0.6;
    let ok = fit(&lenient, &sources, &JointTransportOptions { draws: 10, seed: 1 });
    let ctx = ExecutionContext::for_tests(1);
    let ok2 = fit_joint_bayesian_transport(
        &identified(),
        &lenient,
        &sources,
        Some(&shifted),
        &JointTransportOptions { draws: 10, seed: 1 },
        &ctx,
    )
    .unwrap();
    assert!(ok.is_ok());
    assert!((ok2.diagnostics.overlap.unsupported_mass - 0.5).abs() < 1e-12);
    // Missing law: no target sample, or an empty one.
    let refusal = refusal_of(&model(), &sources, None);
    assert_eq!(refusal.code, "joint_law_required");
    assert_eq!(refusal.detail, DETAIL_MISSING_LAW);
    let empty = TargetData { rows: 0, covariates: vec![vec![]], ..target() };
    assert_eq!(refusal_of(&model(), &sources, Some(&empty)).detail, DETAIL_MISSING_LAW);
}

#[test]
fn x4_prior_data_overlap_bounds_and_source_dependence_refuse() {
    let r1 = raw(30, 0, 1.0, 2.0, 1.5, 0.09);
    let sources = [to_source("s1", &r1, "snap-s1", "s1")];
    for dependence in [SourceDependence::OverlappingUnits, SourceDependence::Unknown] {
        let mut m = model();
        m.dependence = dependence;
        let refusal = refusal_of(&m, &sources, Some(&target()));
        assert_eq!(refusal.code, "sampling_dependence_unknown");
        assert_eq!(refusal.detail, DETAIL_SOURCE_DEPENDENCE);
    }
    // Two sources that own the same unit would be counted twice.
    let r2 = raw(30, 100, 1.0, 2.0, 1.5, 0.09);
    let both = [to_source("s1", &r1, "snap-s1", "dup"), to_source("s2", &r2, "snap-s2", "dup")];
    let refusal = refusal_of(&model(), &both, Some(&target()));
    assert_eq!(refusal.detail, DETAIL_SOURCE_DEPENDENCE);
    assert!(!refusal.failure.overlapping_ids.is_empty());
    // More parameters than the declared bound.
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
        fit_joint_bayesian_transport(&id, &m, &[wide_source], Some(&wide_target), &options(), &ctx)
            .unwrap_err();
    assert_eq!(refusal.code, "invalid_argument");
    assert_eq!(refusal.detail, DETAIL_TOO_MANY_PARAMETERS);
}

// ---------------------------------------------------------------- conflicts and identification

#[test]
fn x4_posterior_conflicting_sources_are_reported_not_silently_pooled() {
    // Two sources with opposite effects and small noise.
    let r1 = raw(60, 0, 1.0, 3.0, 0.0, 0.04);
    let r2 = raw(60, 500, 1.0, -3.0, 0.0, 0.04);
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
    let o = oracle(spec(&raws, false, false, 0.0, 4.0));
    assert_engine_matches_oracle(&f, &o, &raws);
    // Consistent sources are not flagged.
    let r3 = raw(60, 900, 1.0, 3.0, 0.0, 0.04);
    let agreeing = [to_source("s1", &r1, "snap-s1", "s1"), to_source("s3", &r3, "snap-s3", "s3")];
    let g = fit(&model(), &agreeing, &JointTransportOptions { draws: 200, seed: 3 }).unwrap();
    assert!(!g.diagnostics.disagreement.flagged);
    assert!(g.diagnostics.disagreement.max_abs_z < 3.0);
    // A single source has no disagreement to report.
    let one =
        fit(&model(), &conflicting[..1], &JointTransportOptions { draws: 50, seed: 3 }).unwrap();
    assert!(one.diagnostics.disagreement.pairs.is_empty());
}

#[test]
fn x4_posterior_prior_strength_never_changes_identification_status() {
    let r1 = raw(30, 0, 1.0, 2.0, 1.5, 0.09);
    let sources = [to_source("s1", &r1, "snap-s1", "s1")];
    let small = JointTransportOptions { draws: 50, seed: 1 };
    let mut records = Vec::new();
    for var in [1e-8, 1.0, 1e8] {
        let m =
            model_with(VaryingBlock::Intercept, SourceSharing::IndependentVaryingBlocks, 3.0, var);
        records.push(fit(&m, &sources, &small).unwrap().identification);
    }
    assert!(records.windows(2).all(|w| w[0] == w[1]));
    assert_eq!(records[0].status, "identified");
    assert_eq!(records[0].formula, "standardize");
    assert_eq!(records[0].rule, "standardization");
    // An infinitely confident prior cannot rescue an uncertified derivation.
    let tight =
        model_with(VaryingBlock::Intercept, SourceSharing::IndependentVaryingBlocks, 2.0, 1e-12);
    let not_certified = TransportIdentification::NotCertified(NotCertifiedCertificate {
        reason: Arc::from("selection_on_outcome"),
        witness: Arc::from([]),
        message: Arc::from("not identified"),
    });
    let ctx = ExecutionContext::for_tests(1);
    let refusal = fit_joint_bayesian_transport(
        &not_certified,
        &tight,
        &sources,
        Some(&target()),
        &small,
        &ctx,
    )
    .unwrap_err();
    assert_eq!(refusal.code, "transport_not_certified");
}
