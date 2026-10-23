//! Target-native refuters on a whole target vector, and the refusal of refutations whose
//! target is not the claim's.
//!
//! The refit is a complete procedure on simulated data with a known truth: a linear
//! adjustment for `Y = 1 + 0.5·T + X + e` with three treatment levels, whose target is the
//! vector of mean outcomes under each level (a world-valued target).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::execution::AdmissionPool;
use antecedent_core::{CausalRng, ExecutionContext};
use antecedent_kernels::{shuffle, standard_normal, unbiased_index};
use antecedent_stats::{DenseLinearAlgebra, FaerBackend, LeastSquaresWorkspace};
use antecedent_validate::target_native::{
    RefutationTarget, TargetCertificationError, TargetFamily, TargetPerturbation, TargetQuadrant,
    TargetRefit, TargetRefitError, TargetRefutationConfig, TargetRefutationPlan, TargetRefuterKind,
    TargetSuite, certify_target_refutations, refuse_scalar_reports_for_target, run_target_plan,
};
use antecedent_validate::{RefutationReport, ValidationError};

const LEVELS: [f64; 3] = [0.0, 1.0, 2.0];

struct Simulated {
    t: Vec<f64>,
    x: Vec<f64>,
    y: Vec<f64>,
}

fn simulate(n: usize, seed: u64) -> Simulated {
    let mut rng = CausalRng::from_seed(seed);
    let mut sim = Simulated { t: Vec::new(), x: Vec::new(), y: Vec::new() };
    for _ in 0..n {
        let x = standard_normal(&mut rng);
        // Treatment depends on X: confounded, identified by adjusting for X.
        let t = if x + 0.5 * standard_normal(&mut rng) > 0.5 {
            2.0
        } else if x > -0.5 {
            1.0
        } else {
            0.0
        };
        let y = 1.0 + 0.5 * t + x + 0.5 * standard_normal(&mut rng);
        sim.t.push(t);
        sim.x.push(x);
        sim.y.push(y);
    }
    sim
}

/// World values `E[Y | do(T = level)]` from the regression `Y ~ 1 + T + X (+ U)`.
fn world_values(t: &[f64], x: &[f64], y: &[f64], extra: Option<&[f64]>) -> Vec<f64> {
    let n = y.len();
    let mut design = vec![1.0; n];
    design.extend_from_slice(t);
    design.extend_from_slice(x);
    let mut p = 3;
    if let Some(u) = extra {
        design.extend_from_slice(u);
        p += 1;
    }
    let fit =
        FaerBackend.least_squares(&design, n, p, y, &mut LeastSquaresWorkspace::default()).unwrap();
    let b = &fit.coefficients;
    let mean_x = x.iter().sum::<f64>() / n as f64;
    let mean_u = extra.map_or(0.0, |u| u.iter().sum::<f64>() / n as f64);
    LEVELS
        .iter()
        .map(|level| b[0] + b[1] * level + b[2] * mean_x + if p == 4 { b[3] * mean_u } else { 0.0 })
        .collect()
}

struct AdjustmentRefit {
    sim: Simulated,
    target: RefutationTarget,
    original: Vec<f64>,
    /// A defective procedure that reports a dose trend whatever the data say.
    bias: f64,
}

impl AdjustmentRefit {
    fn new(sim: Simulated, bias: f64) -> Self {
        let target = world_target();
        let mut original = world_values(&sim.t, &sim.x, &sim.y, None);
        for (v, level) in original.iter_mut().zip(LEVELS) {
            *v += bias * level;
        }
        Self { sim, target, original, bias }
    }

    fn values(&self, t: &[f64], x: &[f64], y: &[f64], u: Option<&[f64]>) -> Vec<f64> {
        let mut out = world_values(t, x, y, u);
        for (v, level) in out.iter_mut().zip(LEVELS) {
            *v += self.bias * level;
        }
        out
    }

    fn rows(&self, rows: &[usize]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        (
            rows.iter().map(|&i| self.sim.t[i]).collect(),
            rows.iter().map(|&i| self.sim.x[i]).collect(),
            rows.iter().map(|&i| self.sim.y[i]).collect(),
        )
    }
}

fn world_target() -> RefutationTarget {
    RefutationTarget::new(
        TargetFamily::WorldValued,
        TargetQuadrant::STATIC_FREQUENTIST,
        ["do(T=0)", "do(T=1)", "do(T=2)"],
        [3; 32],
    )
    .unwrap()
}

impl TargetRefit for AdjustmentRefit {
    fn target(&self) -> &RefutationTarget {
        &self.target
    }

    fn original(&self) -> &[f64] {
        &self.original
    }

    fn refit(
        &self,
        perturbation: TargetPerturbation,
        rng: &mut CausalRng,
        _ctx: &ExecutionContext,
    ) -> Result<Vec<f64>, TargetRefitError> {
        let n = self.sim.y.len();
        let s = &self.sim;
        Ok(match perturbation {
            TargetPerturbation::PlaceboTreatment { .. } => {
                let mut t = s.t.clone();
                shuffle(rng, &mut t);
                self.values(&t, &s.x, &s.y, None)
            }
            TargetPerturbation::PlaceboOutcome { .. } => {
                let mut y = s.y.clone();
                shuffle(rng, &mut y);
                self.values(&s.t, &s.x, &y, None)
            }
            TargetPerturbation::RandomCommonCause { .. } => {
                let u: Vec<f64> = (0..n).map(|_| standard_normal(rng)).collect();
                self.values(&s.t, &s.x, &s.y, Some(&u))
            }
            TargetPerturbation::Subset { fraction, .. } => {
                let mut rows: Vec<usize> = (0..n).collect();
                shuffle(rng, &mut rows);
                rows.truncate((fraction * n as f64) as usize);
                let (t, x, y) = self.rows(&rows);
                self.values(&t, &x, &y, None)
            }
            TargetPerturbation::Bootstrap { .. } => {
                let rows: Vec<usize> = (0..n).map(|_| unbiased_index(rng, n)).collect();
                let (t, x, y) = self.rows(&rows);
                self.values(&t, &x, &y, None)
            }
            _ => {
                return Err(TargetRefitError::Unsupported {
                    reason: Arc::from("no negative control is declared for this study"),
                    next_attempt: Arc::from("declare a negative-control exposure or outcome"),
                });
            }
        })
    }
}

fn config() -> TargetRefutationConfig {
    TargetRefutationConfig { replicates: 40, ..TargetRefutationConfig::default() }
}

fn report(
    run: &antecedent_validate::target_native::TargetRefutationRun,
    kind: TargetRefuterKind,
) -> &antecedent_validate::TargetRefutationReport {
    run.reports.iter().find(|r| r.kind == kind).unwrap()
}

#[test]
fn a_sound_procedure_passes_the_full_world_valued_suite_and_records_missing_controls() {
    let refit = AdjustmentRefit::new(simulate(600, 11), 0.0);
    // Recovered truth: world values 1 + 0.5 * level + E[X].
    let mean_x = refit.sim.x.iter().sum::<f64>() / 600.0;
    for (v, level) in refit.original.iter().zip(LEVELS) {
        assert!((v - (1.0 + 0.5 * level + mean_x)).abs() < 0.15, "{v} at {level}");
    }
    let plan = TargetRefutationPlan::for_target(world_target(), TargetSuite::Full);
    let run = run_target_plan(&plan, &refit, &config(), &ExecutionContext::for_tests(5)).unwrap();
    for kind in [
        TargetRefuterKind::PlaceboTreatment,
        TargetRefuterKind::PlaceboOutcome,
        TargetRefuterKind::RandomCommonCause,
        TargetRefuterKind::SubsetStability,
        TargetRefuterKind::BootstrapStability,
    ] {
        let r = report(&run, kind);
        assert!(r.passed, "{kind:?}: {:?}", r.failure);
        assert_eq!(r.target.identity(), world_target().identity());
    }
    // The placebo null is tested on the two contrasts between worlds, jointly.
    assert_eq!(report(&run, TargetRefuterKind::PlaceboTreatment).statistic.df, 2);
    assert_eq!(report(&run, TargetRefuterKind::BootstrapStability).statistic.df, 3);
    let refused: Vec<_> = run.refused.iter().map(|r| r.kind).collect();
    assert_eq!(
        refused,
        vec![TargetRefuterKind::NegativeControlExposure, TargetRefuterKind::NegativeControlOutcome]
    );
    assert!(run.refused.iter().all(|r| !r.next_attempt.is_empty()));
}

#[test]
fn a_procedure_that_invents_a_dose_trend_fails_the_placebo_on_the_whole_vector() {
    let refit = AdjustmentRefit::new(simulate(600, 12), 0.4);
    let plan = TargetRefutationPlan::for_target(world_target(), TargetSuite::Placebo);
    let run = run_target_plan(&plan, &refit, &config(), &ExecutionContext::for_tests(5)).unwrap();
    let placebo = report(&run, TargetRefuterKind::PlaceboTreatment);
    assert!(!placebo.passed);
    assert!(placebo.failure.as_deref().unwrap().contains("joint chi-square"));
}

#[test]
fn reports_are_deterministic_and_independent_of_the_thread_count() {
    let plan = TargetRefutationPlan::for_target(world_target(), TargetSuite::Full);
    let a = run_target_plan(
        &plan,
        &AdjustmentRefit::new(simulate(300, 21), 0.0),
        &config(),
        &ExecutionContext::for_tests(9),
    )
    .unwrap();
    let b = run_target_plan(
        &plan,
        &AdjustmentRefit::new(simulate(300, 21), 0.0),
        &config(),
        &ExecutionContext::production(9, 4),
    )
    .unwrap();
    assert_eq!(a, b);
}

#[test]
fn a_tiny_admission_pool_refuses_and_is_restored() {
    let pool = AdmissionPool::fixed(1, 64);
    let before = pool.available();
    let ctx = ExecutionContext::for_tests(1).with_admission(pool.clone());
    let plan = TargetRefutationPlan::for_target(world_target(), TargetSuite::Cheap);
    let error =
        run_target_plan(&plan, &AdjustmentRefit::new(simulate(200, 3), 0.0), &config(), &ctx)
            .unwrap_err();
    assert!(matches!(error, ValidationError::Refused { .. }), "{error:?}");
    assert_eq!(pool.available(), before);
}

#[test]
fn a_scalar_average_effect_refuter_cannot_certify_another_target() {
    let scalar = RefutationReport::new("placebo_treatment", 0.5, 0.01, 0.8, true, true, None, 50);
    let error = refuse_scalar_reports_for_target(&world_target(), &[scalar.clone()]).unwrap_err();
    match &error {
        TargetCertificationError::ScalarRefuterForOtherTarget {
            refuter,
            claim_family,
            next_attempt,
        } => {
            assert_eq!(refuter.as_ref(), "placebo_treatment");
            assert_eq!(*claim_family, TargetFamily::WorldValued);
            assert!(next_attempt.contains("target-native suite"));
        }
        other => panic!("unexpected {other:?}"),
    }
    assert!(error.to_string().contains("cannot certify a world_valued target"));
    // The scalar target itself may cite it.
    let ate = RefutationTarget::scalar_average_effect(TargetQuadrant::STATIC_FREQUENTIST, [3; 32]);
    assert!(refuse_scalar_reports_for_target(&ate, &[scalar]).is_ok());
}

#[test]
fn a_report_on_another_target_is_refused_by_the_claim() {
    let refit = AdjustmentRefit::new(simulate(300, 31), 0.0);
    let plan = TargetRefutationPlan::for_target(world_target(), TargetSuite::Cheap);
    let run = run_target_plan(&plan, &refit, &config(), &ExecutionContext::for_tests(2)).unwrap();
    assert!(certify_target_refutations(&world_target(), &run.reports).is_ok());
    // Same family, different defining identity: a different target.
    let other = RefutationTarget::new(
        TargetFamily::WorldValued,
        TargetQuadrant::STATIC_FREQUENTIST,
        ["do(T=0)", "do(T=1)", "do(T=2)"],
        [4; 32],
    )
    .unwrap();
    let error = certify_target_refutations(&other, &run.reports).unwrap_err();
    assert!(matches!(error, TargetCertificationError::TargetMismatch { .. }));
    assert!(error.to_string().contains("next attempt"));
}

#[test]
fn every_quadrant_has_a_catalogue_for_every_family() {
    for family in [
        TargetFamily::ConditionalEffect,
        TargetFamily::ResponseDerivative,
        TargetFamily::TemporalEffect,
        TargetFamily::Mediation,
        TargetFamily::WorldValued,
        TargetFamily::JointContrast,
    ] {
        for quadrant in [
            TargetQuadrant::STATIC_FREQUENTIST,
            TargetQuadrant::STATIC_BAYESIAN,
            TargetQuadrant::TEMPORAL_FREQUENTIST,
            TargetQuadrant::TEMPORAL_BAYESIAN,
        ] {
            let target = RefutationTarget::new(family, quadrant, ["a", "b"], [1; 32]).unwrap();
            let plan = TargetRefutationPlan::for_target(target, TargetSuite::Full);
            let kinds = plan.refuters();
            assert!(kinds.contains(&TargetRefuterKind::PlaceboTreatment));
            assert_eq!(quadrant.temporal, kinds.contains(&TargetRefuterKind::PreTrend));
            assert_eq!(quadrant.bayesian, kinds.contains(&TargetRefuterKind::PriorSensitivity));
            assert_eq!(quadrant.bayesian, kinds.contains(&TargetRefuterKind::PosteriorPredictive));
            assert_eq!(!quadrant.bayesian, kinds.contains(&TargetRefuterKind::BootstrapStability));
            assert_eq!(
                family == TargetFamily::Mediation,
                kinds.contains(&TargetRefuterKind::SequentialIgnorability)
            );
            assert_eq!(
                family == TargetFamily::ResponseDerivative,
                kinds.contains(&TargetRefuterKind::FiniteDifferenceConsistency)
            );
        }
    }
}
