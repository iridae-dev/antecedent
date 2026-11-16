//! A Gaussian coefficient prior on the outcome nuisance of the cross-fitted AIPW score.
//!
//! The prior replaces the per-arm outcome regression of every fold by its posterior mode,
//! `(X'X + V0^-1)^-1 (X'y + V0^-1 m)`, fitted on the fold's training rows of that arm. The
//! tests check that closed form against an independent hand computation of the whole doubly
//! robust estimate from the published out-of-fold propensities and fold plan, that a correct
//! propensity keeps the estimate consistent under a badly misspecified prior (the prior shapes
//! efficiency, not identification), that a very wide prior returns the unpenalized
//! cross-fitted estimate, and that a prior whose coordinates do not map onto the outcome design
//! is refused with the next attempt.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    reason = "the references index small fixtures and solve 2 x 2 systems by name"
)]

#[path = "oracle_data/table_support.rs"]
mod table_support;

use std::sync::Arc;

use antecedent_core::{
    AssumptionSet, AverageEffectQuery, ExecutionContext, RoleHint, StreamDomain,
};
use antecedent_estimate::{
    AipwAte, AipwWorkspace, DmlAte, DmlScore, HydrateMapping, OUTCOME_PRIOR_ASSUMPTION_ID,
    OutcomePriorSource,
};
use antecedent_expr::{ExprId, IdentifiedEstimand};
use antecedent_kernels::standard_normal;
use antecedent_learn::{LearnerSpec, LinearSpec};
use antecedent_prob::PosteriorQuantityKind;
use table_support::{id, table};

const FOLDS: usize = 4;

struct Draw {
    t: Vec<f64>,
    y: Vec<f64>,
    z: Vec<f64>,
}

/// `z ~ N(0, 1)`, `P(T = 1 | z) = logistic(0.5 z)`, `Y = 1 + 2 T + 0.8 z + 0.5 e`.
fn draw(n: usize, seed: u64) -> Draw {
    let mut rng = ExecutionContext::for_tests(seed).rng.stream_for(StreamDomain::Test, 0x0DD);
    let (mut t, mut y, mut z) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        z[i] = standard_normal(&mut rng);
        t[i] = f64::from(rng.next_f64() < 1.0 / (1.0 + (-0.5 * z[i]).exp()));
        y[i] = 1.0 + 2.0 * t[i] + 0.8 * z[i] + 0.5 * standard_normal(&mut rng);
    }
    Draw { t, y, z }
}

fn setup(draw: &Draw) -> (antecedent_data::TabularData, IdentifiedEstimand, AverageEffectQuery) {
    let data = table(&[
        ("t", RoleHint::TreatmentCandidate, draw.t.clone()),
        ("y", RoleHint::OutcomeCandidate, draw.y.clone()),
        ("z", RoleHint::Context, draw.z.clone()),
    ]);
    let estimand = IdentifiedEstimand::backdoor(
        "backdoor.adjustment",
        Arc::from([id(2)]),
        ExprId::from_raw(0),
    );
    (data, estimand, AverageEffectQuery::binary_ate(id(0), id(1)))
}

/// Source summaries of a fitted outcome posterior: `intercept` and `coef_z2` (the design
/// names of a table whose covariate is the third column), on the conjugate scale of a unit
/// source residual variance so that `V0 = sd^2`.
fn source(mean: [f64; 2], sd: [f64; 2]) -> OutcomePriorSource {
    OutcomePriorSource::new(
        vec![
            PosteriorQuantityKind::Coefficient { index: 0, name: Some(Arc::from("intercept")) },
            PosteriorQuantityKind::Coefficient { index: 1, name: Some(Arc::from("coef_z2")) },
            PosteriorQuantityKind::ResidualVariance,
        ],
        vec![mean[0], mean[1], 1.0],
        vec![sd[0], sd[1], 0.0],
        None,
        HydrateMapping::IdenticalCoefficientSubspace,
        "blake3:reference-source",
    )
    .unwrap()
}

/// Posterior mode of one arm's `[1, z]` regression on `rows`, by an explicit 2 x 2 solve of
/// `(X'X + diag(1/V0)) b = X'y + diag(1/V0) m`.
fn arm_map(rows: &[usize], d: &Draw, mean: [f64; 2], v0: [f64; 2]) -> [f64; 2] {
    let (mut s1, mut sz, mut szz, mut sy, mut szy) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for &i in rows {
        s1 += 1.0;
        sz += d.z[i];
        szz += d.z[i] * d.z[i];
        sy += d.y[i];
        szy += d.z[i] * d.y[i];
    }
    let (a00, a01, a11) = (s1 + 1.0 / v0[0], sz, szz + 1.0 / v0[1]);
    let (b0, b1) = (sy + mean[0] / v0[0], szy + mean[1] / v0[1]);
    let det = a00 * a11 - a01 * a01;
    [(a11 * b0 - a01 * b1) / det, (a00 * b1 - a01 * b0) / det]
}

#[test]
fn the_aipw_point_is_the_hand_computed_doubly_robust_estimate_of_the_fold_map_fits() {
    let (mean, sd) = ([0.3, 1.2], [0.5, 0.4]);
    let d = draw(80, 4_101);
    let (data, estimand, query) = setup(&d);
    let estimator = DmlAte::new().with_folds(FOLDS).with_outcome_prior(source(mean, sd));
    let problem = estimator.prepare(&data, &estimand, &query).unwrap();
    let ctx = ExecutionContext::for_tests(4_101);
    let fit = estimator.fit(&problem, &ctx, AssumptionSet::new()).unwrap();
    assert_hand_computed(&fit, &d, mean, sd);
}

#[test]
fn the_aipw_estimator_route_is_the_hand_computed_doubly_robust_estimate_of_the_fold_map_fits() {
    let (mean, sd) = ([0.3, 1.2], [0.5, 0.4]);
    let d = draw(100, 4_105);
    let (data, estimand, query) = setup(&d);
    let estimator =
        AipwAte::new().with_bootstrap_replicates(0).with_outcome_prior(source(mean, sd));
    let problem = estimator.prepare(&data, &estimand, &query).unwrap();
    let ctx = ExecutionContext::for_tests(4_105);
    let fit =
        estimator.fit(&problem, &mut AipwWorkspace::default(), &ctx, AssumptionSet::new()).unwrap();
    assert_hand_computed(&fit, &d, mean, sd);
    // The bootstrap refits the unregularized nuisances, so a prior with one is refused.
    let boot = AipwAte::new().with_bootstrap_replicates(50).with_outcome_prior(source(mean, sd));
    let problem = boot.prepare(&data, &estimand, &query).unwrap();
    let refused =
        boot.fit(&problem, &mut AipwWorkspace::default(), &ctx, AssumptionSet::new()).unwrap_err();
    assert_eq!(refused.reason_code(), Some("route_not_supported"), "{refused}");
}

#[test]
fn the_aipw_estimator_route_stays_consistent_under_a_misspecified_prior() {
    let reps = 30;
    let estimates: Vec<f64> = (0..reps)
        .map(|rep| {
            let d = draw(2_000, 8_000 + rep);
            let (data, estimand, query) = setup(&d);
            let estimator = AipwAte::new()
                .with_bootstrap_replicates(0)
                .with_outcome_prior(source([10.0, -5.0], [1e-3, 1e-3]));
            let problem = estimator.prepare(&data, &estimand, &query).unwrap();
            estimator
                .fit(
                    &problem,
                    &mut AipwWorkspace::default(),
                    &ExecutionContext::for_tests(8_000 + rep),
                    AssumptionSet::new(),
                )
                .unwrap()
                .ate
        })
        .collect();
    let mean = estimates.iter().sum::<f64>() / reps as f64;
    let var = estimates.iter().map(|e| (e - mean).powi(2)).sum::<f64>() / (reps as f64 - 1.0);
    let se_of_mean = (var / reps as f64).sqrt();
    assert!((mean - 2.0).abs() <= 4.0 * se_of_mean && (mean - 2.0).abs() < 0.5, "{mean}");
}

/// Recompute the whole doubly robust estimate of `fit` from its published out-of-fold
/// propensity and fold plan, with the per-fold arm outcome models solved by an independent 2 x 2
/// system, and compare the point, the per-arm scores, the standard error and the recorded prior.
fn assert_hand_computed(
    fit: &antecedent_estimate::EffectEstimate,
    d: &Draw,
    mean: [f64; 2],
    sd: [f64; 2],
) {
    let n = d.t.len();
    let table = fit.scores.table.as_ref().expect("a score table");
    let v0 = [sd[0] * sd[0], sd[1] * sd[1]];
    // The out-of-fold propensity and the fold plan are the route's own (the propensity is not
    // the object under test); the outcome nuisance and the score are recomputed here.
    let clip = table.propensity_clip;
    let mut phi = vec![0.0; n];
    for i in 0..n {
        let fold = table.fold_ids[i];
        let in_fold = |j: usize| table.fold_ids[j] == fold;
        let control: Vec<usize> = (0..n).filter(|&j| !in_fold(j) && d.t[j] < 0.5).collect();
        let treated: Vec<usize> = (0..n).filter(|&j| !in_fold(j) && d.t[j] > 0.5).collect();
        let (b0, b1) = (arm_map(&control, d, mean, v0), arm_map(&treated, d, mean, v0));
        let (mu0, mu1) = (b0[0] + b0[1] * d.z[i], b1[0] + b1[1] * d.z[i]);
        let mut e = table.propensities[n + i];
        if let Some(c) = clip {
            e = e.clamp(c, 1.0 - c);
        }
        phi[i] =
            (mu1 - mu0) + d.t[i] * (d.y[i] - mu1) / e - (1.0 - d.t[i]) * (d.y[i] - mu0) / (1.0 - e);
        // The published per-arm scores are the same formulas row by row.
        let arm1 = mu1 + d.t[i] * (d.y[i] - mu1) / e;
        let arm0 = mu0 + (1.0 - d.t[i]) * (d.y[i] - mu0) / (1.0 - e);
        assert!((table.scores[n + i] - arm1).abs() <= 1e-9, "arm-1 score row {i}");
        assert!((table.scores[i] - arm0).abs() <= 1e-9, "arm-0 score row {i}");
    }
    let expected = phi.iter().sum::<f64>() / n as f64;
    assert!((fit.ate - expected).abs() <= 1e-9, "{} vs {expected}", fit.ate);
    // The interval keeps its form: the standard error is that of the score's influence, equal to
    // the iid standard error of the hand-computed scores up to the variance denominator.
    let centred: f64 = phi.iter().map(|v| (v - expected).powi(2)).sum::<f64>();
    let se = (centred / (n as f64 * (n as f64 - 1.0))).sqrt();
    let published = fit.uncertainty.analytic.se;
    assert!((published - se).abs() <= se / n as f64, "{published} vs {se}");
    // The prior and its source are recorded as an assumption that creates no identification.
    let assumptions = &fit.provenance.assumptions.entries;
    assert!(
        assumptions.iter().any(|record| matches!(
            &record.assumption,
            antecedent_core::Assumption::PriorRestriction(prior)
                if prior.id.as_ref() == OUTCOME_PRIOR_ASSUMPTION_ID
                    && prior.description.contains("blake3:reference-source")
        )),
        "{assumptions:?}"
    );
    assert!(
        table
            .nuisance_provenance
            .contains("outcome=gaussian_prior_linear;prior=blake3:reference-source"),
        "{}",
        table.nuisance_provenance
    );
}

#[test]
fn the_prior_is_recorded_in_the_assumptions_and_the_score_table_provenance() {
    let d = draw(60, 4_106);
    let (data, estimand, query) = setup(&d);
    let estimator =
        DmlAte::new().with_folds(FOLDS).with_outcome_prior(source([0.0, 1.0], [1.0, 1.0]));
    let problem = estimator.prepare(&data, &estimand, &query).unwrap();
    let fit =
        estimator.fit(&problem, &ExecutionContext::for_tests(4_106), AssumptionSet::new()).unwrap();
    assert!(fit.provenance.assumptions.entries.iter().any(|record| matches!(
        &record.assumption,
        antecedent_core::Assumption::PriorRestriction(prior)
            if prior.id.as_ref() == OUTCOME_PRIOR_ASSUMPTION_ID
    )));
    let table = fit.scores.table.as_ref().expect("a score table");
    assert!(
        table.nuisance_provenance.contains(
            "outcome=gaussian_prior_linear;prior=blake3:reference-source;mapping=identical"
        ),
        "{}",
        table.nuisance_provenance
    );
}

#[test]
fn a_correct_propensity_keeps_the_estimate_consistent_under_a_badly_misspecified_prior() {
    // A prior pinned to the line 10 - 5 z for both arms: the outcome nuisance alone would give a
    // zero effect, yet the propensity is the true logistic, so the correction removes the bias.
    let reps = 40;
    let mut estimates = Vec::new();
    for rep in 0..reps {
        let d = draw(2_000, 7_000 + rep);
        let (data, estimand, query) = setup(&d);
        let estimator =
            DmlAte::new().with_folds(FOLDS).with_outcome_prior(source([10.0, -5.0], [1e-3, 1e-3]));
        let problem = estimator.prepare(&data, &estimand, &query).unwrap();
        let fit = estimator
            .fit(&problem, &ExecutionContext::for_tests(7_000 + rep), AssumptionSet::new())
            .unwrap();
        estimates.push(fit.ate);
    }
    let mean = estimates.iter().sum::<f64>() / reps as f64;
    let var = estimates.iter().map(|e| (e - mean).powi(2)).sum::<f64>() / (reps as f64 - 1.0);
    let se_of_mean = (var / reps as f64).sqrt();
    assert!(
        (mean - 2.0).abs() <= 4.0 * se_of_mean && (mean - 2.0).abs() < 0.5,
        "mean {mean} (se {se_of_mean}) against the effect 2; the pinned outcome model alone gives 0"
    );
}

#[test]
fn a_very_wide_prior_returns_the_unpenalized_cross_fitted_estimate() {
    let d = draw(300, 4_103);
    let (data, estimand, query) = setup(&d);
    let ctx = ExecutionContext::for_tests(4_103);
    let plain = DmlAte::new().with_folds(FOLDS).with_outcome(LearnerSpec::Linear(LinearSpec {}));
    let reference = plain
        .fit(&plain.prepare(&data, &estimand, &query).unwrap(), &ctx, AssumptionSet::new())
        .unwrap();
    let wide = DmlAte::new().with_folds(FOLDS).with_outcome_prior(source([3.0, -7.0], [1e5, 1e5]));
    let fit = wide
        .fit(&wide.prepare(&data, &estimand, &query).unwrap(), &ctx, AssumptionSet::new())
        .unwrap();
    assert!((fit.ate - reference.ate).abs() <= 1e-6, "{} vs {}", fit.ate, reference.ate);
}

#[test]
fn a_prior_that_does_not_map_onto_the_outcome_design_is_refused_with_the_next_attempt() {
    let d = draw(80, 4_104);
    let (data, estimand, query) = setup(&d);
    let ctx = ExecutionContext::for_tests(4_104);
    // The source carries a treatment coefficient the per-arm outcome regression does not have.
    let three = OutcomePriorSource::new(
        vec![
            PosteriorQuantityKind::Coefficient { index: 0, name: Some(Arc::from("intercept")) },
            PosteriorQuantityKind::Coefficient { index: 1, name: Some(Arc::from("coef_t0")) },
            PosteriorQuantityKind::Coefficient { index: 2, name: Some(Arc::from("coef_z2")) },
        ],
        vec![0.0, 2.0, 0.8],
        vec![1.0, 1.0, 1.0],
        None,
        HydrateMapping::IdenticalCoefficientSubspace,
        "blake3:three",
    )
    .unwrap();
    let estimator = DmlAte::new().with_folds(FOLDS).with_outcome_prior(three);
    let problem = estimator.prepare(&data, &estimand, &query).unwrap();
    let refused = estimator.fit(&problem, &ctx, AssumptionSet::new()).unwrap_err();
    assert_eq!(refused.reason_code(), Some("quantity_semantics_mismatch"), "{refused}");
    assert!(refused.to_string().contains("PriorMapping.named_parameters"), "{refused}");
    // A named pairing of the two shared coordinates maps.
    let paired = OutcomePriorSource::new(
        vec![
            PosteriorQuantityKind::Coefficient { index: 0, name: Some(Arc::from("intercept")) },
            PosteriorQuantityKind::Coefficient { index: 1, name: Some(Arc::from("coef_t0")) },
            PosteriorQuantityKind::Coefficient { index: 2, name: Some(Arc::from("coef_z2")) },
        ],
        vec![0.0, 2.0, 0.8],
        vec![1.0, 1.0, 1.0],
        None,
        HydrateMapping::NamedParameters {
            pairs: vec![
                ("intercept".into(), "intercept".into()),
                ("coef_z2".into(), "coef_z2".into()),
            ],
        },
        "blake3:three",
    )
    .unwrap();
    let estimator = DmlAte::new().with_folds(FOLDS).with_outcome_prior(paired);
    let problem = estimator.prepare(&data, &estimand, &query).unwrap();
    estimator.fit(&problem, &ctx, AssumptionSet::new()).unwrap();
    // The partially linear score has no per-arm outcome regression to regularize.
    let plr = DmlAte::new()
        .with_folds(FOLDS)
        .with_score(DmlScore::PartiallyLinear)
        .with_outcome_prior(source([0.0, 0.0], [1.0, 1.0]));
    let problem = plr.prepare(&data, &estimand, &query).unwrap();
    let refused = plr.fit(&problem, &ctx, AssumptionSet::new()).unwrap_err();
    assert_eq!(refused.reason_code(), Some("route_not_supported"), "{refused}");
}

// -- Composed external priors with named coordinates ------------------------------------------

mod composed {
    use antecedent_prob::{
        ExternalPriorSource, ExternalPriorWeight, GaussianCoefficientPrior, KernelFamily, PriorSet,
        PriorSpec, TransportKernel, TransportPolicy, WeightedTarget,
        compose_with_weighted_target_transport,
    };

    use super::{
        Arc, AssumptionSet, DmlAte, Draw, ExecutionContext, FOLDS, OutcomePriorSource, arm_map,
        draw, setup,
    };

    fn gaussian(mean: [f64; 3], variance: [f64; 3]) -> PriorSet {
        let mut prior = PriorSet::new();
        prior.push(PriorSpec::GaussianCoefficients(GaussianCoefficientPrior {
            mean: Arc::from(mean.to_vec()),
            variance: Arc::from(variance.to_vec()),
        }));
        prior
    }

    fn names(items: &[&str]) -> Vec<Arc<str>> {
        items.iter().map(|name| Arc::from(*name)).collect()
    }

    /// Two named sources over `[intercept, coef_z2, coef_t0]` transported toward a Gaussian
    /// kernel target (the transported coordinate is `coef_z2`) and composed by the power prior.
    fn kernel_weighted() -> antecedent_prob::ComposedPrior {
        let source = |id: &str, mean: [f64; 3], alpha: f64| ExternalPriorSource {
            id: Arc::from(id),
            prior: gaussian(mean, [0.4, 0.3, 0.5]),
            weight: ExternalPriorWeight::power(alpha).unwrap(),
            ess: None,
        };
        let sources = [source("a", [0.2, 1.0, 1.8], 1.0), source("b", [0.5, 0.6, 2.2], 0.5)];
        let rows = 40;
        let effects: Vec<f64> = (0..rows).map(|i| 0.6 + 0.01 * i as f64).collect();
        let distances: Vec<f64> = (0..rows).map(|i| i as f64 / rows as f64).collect();
        let kernel = TransportKernel::new(KernelFamily::Gaussian, 1.0).unwrap();
        let target = WeightedTarget::from_kernel(effects, &distances, kernel).unwrap();
        let baseline = gaussian([0.0; 3], [100.0; 3]);
        let (composed, _) = compose_with_weighted_target_transport(
            &sources,
            &baseline,
            TransportPolicy::InvariantConditionalOutcome,
            &target,
            Some(1),
        )
        .unwrap();
        composed.with_coefficient_names(names(&["intercept", "coef_z2", "coef_t0"])).unwrap()
    }

    /// The doubly robust estimate recomputed from the published propensities with the per-fold
    /// arm models solved by the independent 2 x 2 system of the composed prior's two mapped
    /// coordinates.
    fn hand_computed(
        fit: &antecedent_estimate::EffectEstimate,
        d: &Draw,
        mean: [f64; 2],
        v0: [f64; 2],
    ) {
        let n = d.t.len();
        let table = fit.scores.table.as_ref().expect("a score table");
        let mut phi = vec![0.0; n];
        for i in 0..n {
            let fold = table.fold_ids[i];
            let in_fold = |j: usize| table.fold_ids[j] == fold;
            let control: Vec<usize> = (0..n).filter(|&j| !in_fold(j) && d.t[j] < 0.5).collect();
            let treated: Vec<usize> = (0..n).filter(|&j| !in_fold(j) && d.t[j] > 0.5).collect();
            let (b0, b1) = (arm_map(&control, d, mean, v0), arm_map(&treated, d, mean, v0));
            let (mu0, mu1) = (b0[0] + b0[1] * d.z[i], b1[0] + b1[1] * d.z[i]);
            let mut e = table.propensities[n + i];
            if let Some(c) = table.propensity_clip {
                e = e.clamp(c, 1.0 - c);
            }
            phi[i] = (mu1 - mu0) + d.t[i] * (d.y[i] - mu1) / e
                - (1.0 - d.t[i]) * (d.y[i] - mu0) / (1.0 - e);
        }
        let expected = phi.iter().sum::<f64>() / n as f64;
        assert!((fit.ate - expected).abs() <= 1e-9, "{} vs {expected}", fit.ate);
    }

    #[test]
    fn a_kernel_weighted_composed_prior_gives_the_hand_computed_map_nuisance_estimate() {
        let composed = kernel_weighted();
        let coef = composed.prior.gaussian_coefficients().unwrap();
        let (mean, v0) = ([coef.mean[0], coef.mean[1]], [coef.variance[0], coef.variance[1]]);
        let d = draw(80, 4_201);
        let (data, estimand, query) = setup(&d);
        let source = OutcomePriorSource::from_composed(
            composed.coefficient_names.clone(),
            composed.prior.clone(),
            "composed:test",
        )
        .unwrap();
        let estimator = DmlAte::new().with_folds(FOLDS).with_outcome_prior(source);
        let problem = estimator.prepare(&data, &estimand, &query).unwrap();
        let fit = estimator
            .fit(&problem, &ExecutionContext::for_tests(4_201), AssumptionSet::new())
            .unwrap();
        hand_computed(&fit, &d, mean, v0);
        // The treatment coordinate has no column in the per-arm regression: it is dropped, and
        // the name mapping is recorded beside the kernel and the composition.
        let ids: Vec<String> = fit
            .provenance
            .assumptions
            .entries
            .iter()
            .filter_map(|record| match &record.assumption {
                antecedent_core::Assumption::PriorRestriction(prior) => Some(prior.id.to_string()),
                _ => None,
            })
            .collect();
        for expected in ["external_prior_name_mapping", "external_composed_prior"] {
            assert!(ids.iter().any(|id| id == expected), "{expected} in {ids:?}");
        }
        let table = fit.scores.table.as_ref().unwrap();
        assert!(
            table.nuisance_provenance.contains("mapping=composed:intercept,coef_z2,coef_t0"),
            "{}",
            table.nuisance_provenance
        );
    }

    #[test]
    fn an_unnamed_composed_prior_is_positional_and_a_prior_with_no_matching_names_is_refused() {
        let d = draw(60, 4_202);
        let (data, estimand, query) = setup(&d);
        let ctx = ExecutionContext::for_tests(4_202);
        let two = {
            let mut prior = PriorSet::new();
            prior.push(PriorSpec::GaussianCoefficients(
                GaussianCoefficientPrior::shared(2, 0.5, 0.4).unwrap(),
            ));
            prior
        };
        let positional =
            OutcomePriorSource::from_composed(None, two.clone(), "composed:two").unwrap();
        let estimator = DmlAte::new().with_folds(FOLDS).with_outcome_prior(positional);
        let problem = estimator.prepare(&data, &estimand, &query).unwrap();
        let fit = estimator.fit(&problem, &ctx, AssumptionSet::new()).unwrap();
        hand_computed(&fit, &d, [0.5, 0.5], [0.4, 0.4]);
        let stranger = OutcomePriorSource::from_composed(
            Some(Arc::from(names(&["alpha", "beta"]))),
            two,
            "composed:strangers",
        )
        .unwrap();
        let estimator = DmlAte::new().with_folds(FOLDS).with_outcome_prior(stranger);
        let problem = estimator.prepare(&data, &estimand, &query).unwrap();
        let refused = estimator.fit(&problem, &ctx, AssumptionSet::new()).unwrap_err();
        assert_eq!(refused.reason_code(), Some("quantity_semantics_mismatch"), "{refused}");
        assert!(refused.to_string().contains("coefficient_names"), "{refused}");
    }
}
