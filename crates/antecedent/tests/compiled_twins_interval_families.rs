//! The compiled measured-interval families reproduce the deprecated direct intervals, and each
//! reports an independently known truth.
//!
//! Five program families (`CountFittedInterval`, `CohortMeanInterval`, `AdaptiveResultInterval`,
//! `ChoiceResultInterval`, `TemporalDependentInterval`) run the measured consumer of their
//! deprecated direct entry. Each twin test asserts that the compiled run carries the direct
//! entry's measured inference bit for bit (every scalar's point, lower and upper bound and level
//! by `to_bits`, its calibration slot, the whole report and the exported envelope), that a
//! refusal is the same refusal, and that the run names the measured route the direct entry's
//! coverage is recorded under.
//!
//! The independent checks never compare the route to itself:
//!
//! * count-fitted: a single fitted node with `0` and `39` counts has the posterior
//!   `Beta(40, 1)` for `P(Y = 1)`, whose mean is `40 / 41` and whose equal-tailed quantiles are
//!   `p^(1/40)` in closed form;
//! * cohort mean: rows come in pairs on one design point with opposite noise, so ordinary least
//!   squares recovers the generating coefficients exactly and the cohort-mean point is the
//!   cohort's mean of the true effect `1 + 0.5 x`; the Student-t interval is symmetric around it;
//! * adaptive and choice: the posterior of the second arm of a Thompson trial and of the first
//!   option's share of a choice study are Beta laws whose parameters are recounted from the
//!   retained history here; their mean and equal-tailed quantiles are computed by Simpson
//!   integration of the Beta density and bisection, without the library;
//! * temporal: the marginalized two-step response of an exact-count dynamic SCM panel is the
//!   hand-enumerated `0.3 * 0.33 + 0.7 * 0.46 = 0.421`.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    deprecated,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::float_cmp,
    reason = "the deprecated direct intervals are the twins under test; fixture counts and indices \
              are small and bits are compared exactly"
)]

use antecedent::Study;
use antecedent::analysis::compiled_measured_interval::{MeasuredInference, MeasuredIntervalRun};
use antecedent_core::ExecutionContext;

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(5)
}

/// The compiled run carries the direct measured inference bit for bit.
fn assert_twin(direct: &MeasuredInference, run: &MeasuredIntervalRun) {
    let (left, right) = (direct.report(), run.measured.report());
    assert_eq!(left.scalars.len(), right.scalars.len());
    for (a, b) in left.scalars.iter().zip(&right.scalars) {
        assert_eq!(a.name, b.name);
        assert_eq!(a.point.to_bits(), b.point.to_bits(), "point of {}", a.name);
        assert_eq!(a.lower.to_bits(), b.lower.to_bits(), "lower bound of {}", a.name);
        assert_eq!(a.upper.to_bits(), b.upper.to_bits(), "upper bound of {}", a.name);
        assert_eq!(a.level.to_bits(), b.level.to_bits(), "level of {}", a.name);
        assert_eq!(a.calibration, b.calibration, "calibration slot of {}", a.name);
    }
    assert_eq!(left, right, "the whole report");
    assert_eq!(direct.export(), run.measured.export(), "the exported envelope");
    assert_eq!(direct.source_artifact(), run.measured.source_artifact());
    // The coverage that annotates the direct interval is recorded under this route.
    assert_eq!(left.route, run.calibration_route);
}

/// `ln Gamma(n)` of a positive integer.
fn ln_gamma_integer(n: u32) -> f64 {
    (1..n).map(|k| f64::from(k).ln()).sum()
}

/// `P(X <= x)` for `X ~ Beta(a, b)` with integer `a, b >= 1`, by composite Simpson integration of
/// the density.
fn beta_cdf(a: u32, b: u32, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let ln_beta = ln_gamma_integer(a) + ln_gamma_integer(b) - ln_gamma_integer(a + b);
    let density = |t: f64| {
        if t <= 0.0 || t >= 1.0 {
            let at_zero = if a == 1 { 1.0 } else { 0.0 };
            let at_one = if b == 1 { 1.0 } else { 0.0 };
            let edge = if t <= 0.0 { at_zero } else { at_one };
            return edge * (-ln_beta).exp();
        }
        (f64::from(a - 1) * t.ln() + f64::from(b - 1) * (1.0 - t).ln() - ln_beta).exp()
    };
    let steps = 20_000_u32;
    let h = x / f64::from(steps);
    let mut sum = density(0.0) + density(x);
    for i in 1..steps {
        let weight = if i % 2 == 1 { 4.0 } else { 2.0 };
        sum += weight * density(f64::from(i) * h);
    }
    sum * h / 3.0
}

/// The `p` quantile of `Beta(a, b)` by bisection on the Simpson CDF.
fn beta_quantile(a: u32, b: u32, p: f64) -> f64 {
    let (mut low, mut high) = (0.0_f64, 1.0_f64);
    for _ in 0..60 {
        let mid = 0.5 * (low + high);
        if beta_cdf(a, b, mid) < p {
            low = mid;
        } else {
            high = mid;
        }
    }
    0.5 * (low + high)
}

#[test]
fn the_independent_beta_quantiles_reproduce_closed_forms() {
    // Beta(a, 1) has quantile p^(1/a); Beta(1, b) has 1 - (1 - p)^(1/b).
    for p in [0.025, 0.5, 0.975] {
        assert!((beta_quantile(5, 1, p) - p.powf(0.2)).abs() < 1e-9);
        assert!((beta_quantile(1, 3, p) - (1.0 - (1.0 - p).powf(1.0 / 3.0))).abs() < 1e-9);
    }
}

mod count_fitted {
    use super::{Study, assert_twin, ctx};
    use antecedent::analysis::count_fitted_observation::{
        CountFittedNodeWire, CountFittedObservation, ROUTE,
    };

    fn levels() -> Vec<String> {
        vec!["0".into(), "1".into()]
    }

    /// `Z -> Y`, 100 rows: the fixture of the facade tests.
    fn fitted() -> CountFittedObservation {
        let nodes = vec![
            CountFittedNodeWire {
                name: "Z".into(),
                levels: levels(),
                parents: vec![],
                counts: Some(vec![60, 40]),
                declared_bits: None,
            },
            CountFittedNodeWire {
                name: "Y".into(),
                levels: levels(),
                parents: vec![0],
                counts: Some(vec![48, 12, 10, 30]),
                declared_bits: None,
            },
        ];
        CountFittedObservation::new(nodes, 100, "Y", "1", &[("Z".into(), "1".into())], Some(7))
            .unwrap()
    }

    #[test]
    fn the_compiled_count_fitted_interval_equals_the_direct_interval() {
        for level in [0.95, 0.9, 0.5] {
            let direct = fitted().interval(level, &ctx()).expect("reported");
            let compiled = Study::compile_count_fitted_interval(fitted(), level).unwrap();
            let run = compiled.execute(&ctx()).expect("reported");
            assert_twin(&direct, &run);
            assert_eq!(run.calibration_route, ROUTE);
            let calibrated = fitted().calibrated_probability(level, &ctx()).unwrap();
            assert_twin(&calibrated, &run);
        }
        // The draw count is part of the source; a changed count changes the program.
        let more = fitted().with_draws(4000).unwrap();
        let direct = more.interval(0.95, &ctx()).unwrap();
        let run =
            Study::compile_count_fitted_interval(more, 0.95).unwrap().execute(&ctx()).unwrap();
        assert_twin(&direct, &run);
    }

    #[test]
    fn a_refused_level_refuses_identically_through_the_compiled_route() {
        let direct = fitted().interval(1.5, &ctx()).unwrap_err();
        let compiled = Study::compile_count_fitted_interval(fitted(), 1.5).unwrap();
        assert_eq!(compiled.execute(&ctx()).unwrap_err(), direct);
    }

    #[test]
    fn the_count_fitted_program_identity_binds_the_counts_and_the_level() {
        let identity = |observation: CountFittedObservation, level: f64| {
            Study::compile_count_fitted_interval(observation, level).unwrap().lowered().identity()
        };
        assert_eq!(identity(fitted(), 0.95), identity(fitted(), 0.95));
        assert_ne!(identity(fitted(), 0.95), identity(fitted(), 0.9));
        assert_ne!(identity(fitted(), 0.95), identity(fitted().with_draws(4000).unwrap(), 0.95));
    }

    #[test]
    fn the_compiled_count_fitted_interval_recovers_the_closed_form_beta_posterior() {
        // One fitted node, 39 rows all at level 1: P(Y = 1) ~ Beta(40, 1) under the uniform
        // prior, mean 40/41, quantile p^(1/40).
        let node = CountFittedNodeWire {
            name: "Y".into(),
            levels: levels(),
            parents: vec![],
            counts: Some(vec![0, 39]),
            declared_bits: None,
        };
        let observation = CountFittedObservation::new(vec![node], 39, "Y", "1", &[], Some(11))
            .unwrap()
            .with_draws(20_000)
            .unwrap();
        let run = Study::compile_count_fitted_interval(observation, 0.95)
            .unwrap()
            .execute(&ctx())
            .expect("reported");
        let scalar = &run.measured.report().scalars[0];
        assert!((scalar.point - 40.0 / 41.0).abs() < 1e-12, "point {}", scalar.point);
        let lower = 0.025_f64.powf(1.0 / 40.0);
        let upper = 0.975_f64.powf(1.0 / 40.0);
        // Monte Carlo standard errors of the 20000-draw quantiles: about 1e-3 and 3e-5.
        assert!((scalar.lower - lower).abs() < 6e-3, "lower {} vs {lower}", scalar.lower);
        assert!((scalar.upper - upper).abs() < 3e-4, "upper {} vs {upper}", scalar.upper);
    }
}

mod cohort_mean {
    use super::{Study, assert_twin, ctx};
    use antecedent::analysis::cohort_decision::{
        CATE_FUNCTIONAL_ID, CohortBinding, CohortColumn, CohortDecisionSource, CohortOptions,
        ContrastDeclaration, FittedEffectModel, NativeCohortFit, fit_cohort_effect,
        measured_cohort_interval,
    };
    use antecedent_io::quantity_wire::ScientificQuantityWire;

    fn wire(id: &str, role: &str, functional: &str, regime: &str) -> ScientificQuantityWire {
        ScientificQuantityWire {
            version: 1,
            variable_id: id.into(),
            variable_name: id.into(),
            role: role.into(),
            units: "unit".into(),
            population_id: "target".into(),
            regime_id: regime.into(),
            horizon: 0,
            functional_id: functional.into(),
            conditioning: vec![],
            transform_id: "identity".into(),
        }
    }

    fn lcg(state: &mut u64) -> f64 {
        *state =
            state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        ((*state >> 11) as f64 + 0.5) / 9_007_199_254_740_992.0
    }

    /// The true conditional effect `1 + 0.5 x`.
    fn effect(x: f64) -> f64 {
        1.0 + 0.5 * x
    }

    /// `pairs` design points, each observed twice with opposite noise: the residuals are
    /// orthogonal to every design column, so ordinary least squares returns the generating
    /// coefficients.
    fn data(pairs: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut s = 29_u64;
        let (mut x, mut t, mut y) = (vec![], vec![], vec![]);
        for i in 0..pairs {
            let xi = 4.0 * lcg(&mut s) - 2.0;
            let ti = f64::from(u8::from(i % 2 == 0));
            let mean = 2.0 + xi + ti * effect(xi);
            let noise = lcg(&mut s) - 0.5;
            for sign in [1.0, -1.0] {
                x.push(xi);
                t.push(ti);
                y.push(mean + sign * noise);
            }
        }
        (x, t, y)
    }

    fn native_fit(x: &[f64], t: &[f64], y: &[f64]) -> NativeCohortFit {
        let response = wire(
            "y",
            "outcome",
            CATE_FUNCTIONAL_ID,
            &ContrastDeclaration::regime_of("t", 1.0, 0.0),
        );
        fit_cohort_effect(y, t, &[x], &response, &[wire("x", "covariate", "mean", "observed")])
            .unwrap()
    }

    fn source_of(native: &NativeCohortFit, cohort_x: &[f64]) -> CohortDecisionSource {
        let model = FittedEffectModel::from_compact_export(native.export()).unwrap();
        let cohort = CohortBinding::new(
            "cohort",
            "target",
            (0..cohort_x.len()).map(|i| format!("r{i}")).collect(),
            vec!["x".into()],
            vec![CohortColumn::Numeric(cohort_x.to_vec())],
            CohortBinding::uniform_weights(cohort_x.len()).unwrap(),
        )
        .unwrap();
        model.decision_source(&cohort, "unit", &CohortOptions::default(), &ctx()).unwrap()
    }

    /// Every other design point: one row of each pair.
    fn cohort_x(x: &[f64]) -> Vec<f64> {
        x.iter().step_by(2).take(100).copied().collect()
    }

    #[test]
    fn the_compiled_cohort_interval_equals_the_direct_interval() {
        let (x, t, y) = data(200);
        let native = native_fit(&x, &t, &y);
        let cohort = cohort_x(&x);
        let source = source_of(&native, &cohort);
        for level in [0.95, 0.9] {
            let direct = measured_cohort_interval(&native, &source, level, &ctx()).unwrap();
            let run = Study::compile_cohort_interval(native.clone(), source.clone(), level)
                .unwrap()
                .execute(&ctx())
                .unwrap();
            assert_twin(&direct, &run);
        }
        // An invalid level refuses identically.
        let direct = measured_cohort_interval(&native, &source, 1.0, &ctx()).unwrap_err();
        let compiled = Study::compile_cohort_interval(native, source, 1.0).unwrap();
        assert_eq!(format!("{:?}", compiled.execute(&ctx()).unwrap_err()), format!("{direct:?}"));
    }

    #[test]
    fn the_cohort_program_identity_binds_the_fit_the_cohort_and_the_level() {
        let (x, t, y) = data(200);
        let native = native_fit(&x, &t, &y);
        let source = source_of(&native, &cohort_x(&x));
        let identity = |level: f64| {
            Study::compile_cohort_interval(native.clone(), source.clone(), level)
                .unwrap()
                .lowered()
                .identity()
        };
        assert_eq!(identity(0.95), identity(0.95));
        assert_ne!(identity(0.95), identity(0.9));
        let other_cohort = source_of(&native, &cohort_x(&x)[..50]);
        let other = Study::compile_cohort_interval(native.clone(), other_cohort, 0.95).unwrap();
        assert_ne!(identity(0.95), other.lowered().identity());
    }

    #[test]
    fn the_compiled_cohort_interval_is_centred_on_the_true_cohort_mean_effect() {
        let (x, t, y) = data(200);
        let native = native_fit(&x, &t, &y);
        let cohort = cohort_x(&x);
        let truth = cohort.iter().map(|&xi| effect(xi)).sum::<f64>() / cohort.len() as f64;
        let run = Study::compile_cohort_interval(native.clone(), source_of(&native, &cohort), 0.95)
            .unwrap()
            .execute(&ctx())
            .unwrap();
        let scalar = &run.measured.report().scalars[0];
        assert!((scalar.point - truth).abs() < 1e-9, "point {} vs truth {truth}", scalar.point);
        // A symmetric Student-t band around the point that contains the truth.
        let (below, above) = (scalar.point - scalar.lower, scalar.upper - scalar.point);
        assert!(below > 0.0 && (below - above).abs() < 1e-12, "{below} vs {above}");
        assert!(scalar.lower < truth && truth < scalar.upper);
    }
}

/// Beta-Bernoulli Thompson sampling with uniform priors, as in the adaptive facade tests.
mod adaptive {
    use super::{Study, assert_twin, beta_quantile, ctx};
    use antecedent::analysis::adaptive_inference::{
        AdaptiveDesignDeclaration, AdaptiveInferenceSession, AdaptiveRequest, AssignmentFamily,
        DelayMechanism, InterimAnalysis, MonitoringStrategy, OutcomeAvailability, OutcomeStructure,
        PolicyEpoch, ProcedureCell, StoppingRuleDeclaration, StoppingRuleKind, StoppingSpec,
    };
    use antecedent_core::{CausalRng, SemanticDigest};
    use antecedent_stats::adaptive_inference::{BetaPosterior, best_arm_probabilities};

    fn thompson_trial(
        truth: [f64; 2],
        total: usize,
        sizes: &[usize],
        seed: u64,
    ) -> AdaptiveDesignDeclaration {
        let mut rng = CausalRng::from_seed(seed);
        let mut counts = [[0_u64; 2]; 2];
        let (mut arms, mut rows, mut outcomes) = (Vec::new(), Vec::new(), Vec::new());
        for _ in 0..total {
            let posteriors = [0, 1]
                .map(|a| BetaPosterior::from_counts(1.0, 1.0, counts[a][0], counts[a][1]).unwrap());
            let best = best_arm_probabilities(&posteriors, &|| false).unwrap();
            let sum = best[0] + best[1];
            let e = [best[0] / sum, best[1] / sum];
            let arm = usize::from(rng.next_f64() >= e[0]);
            let success = rng.next_f64() < truth[arm];
            counts[arm][usize::from(!success)] += 1;
            arms.push(u32::try_from(arm).unwrap());
            rows.push(e.to_vec());
            outcomes.push(vec![f64::from(u8::from(success))]);
        }
        let looks = sizes
            .iter()
            .map(|&n| InterimAnalysis {
                units_enrolled: u32::try_from(n).unwrap(),
                information_fraction: n as f64 / total as f64,
            })
            .collect();
        let available_at_look = (0..total)
            .map(|unit| Some(u32::try_from(sizes.iter().position(|&n| unit < n).unwrap()).unwrap()))
            .collect();
        AdaptiveDesignDeclaration {
            label: "thompson".to_owned(),
            cell: ProcedureCell {
                assignment: AssignmentFamily::ThompsonBetaBernoulli,
                outcome: OutcomeStructure::SingleOutcome,
                monitoring: MonitoringStrategy::FixedInterimLooks,
                stopping: StoppingRuleKind::Efficacy,
            },
            arm_labels: vec!["control".to_owned(), "treatment".to_owned()],
            assigned_arm: arms.clone(),
            policy_history: Some(vec![PolicyEpoch {
                first_unit: 0,
                policy_identity: SemanticDigest::from_bytes([8; 32]),
            }]),
            assignment_probabilities: Some(rows),
            exposure: Some(arms),
            availability: Some(OutcomeAvailability {
                available_at_look,
                delay: DelayMechanism::Immediate,
            }),
            outcomes,
            interim_analyses: Some(looks),
            stopping_rule: Some(StoppingRuleDeclaration {
                kind: StoppingRuleKind::Efficacy,
                spec: StoppingSpec::PosteriorProbabilityBest { threshold: 0.95 },
                ended_by_rule: false,
            }),
            probability_floor: 1e-6,
        }
    }

    fn session(declaration: AdaptiveDesignDeclaration) -> AdaptiveInferenceSession {
        AdaptiveInferenceSession::prepare(declaration).expect("complete contract")
    }

    fn posterior() -> AdaptiveRequest {
        AdaptiveRequest::BayesianPosterior { look: 1, prior_alpha: 1.0, prior_beta: 1.0 }
    }

    #[test]
    fn the_compiled_adaptive_interval_equals_the_direct_interval() {
        let session = session(thompson_trial([0.3, 0.5], 200, &[100, 200], 5));
        let other_prior =
            AdaptiveRequest::BayesianPosterior { look: 1, prior_alpha: 2.0, prior_beta: 3.0 };
        for (request, id) in [(posterior(), "thompson"), (other_prior, "other-prior")] {
            let direct = session.measured_interval(&request, id, &ctx()).expect("reported");
            let run = Study::compile_adaptive_interval(&session, request, id)
                .unwrap()
                .execute(&ctx())
                .expect("reported");
            assert_twin(&direct, &run);
            assert_eq!(run.calibration_route, "adaptive_interval");
        }
        // A paradigm the cell does not support refuses identically.
        let weighting = AdaptiveRequest::AdaptiveWeighting {
            look: 1,
            plus_arm: 1,
            minus_arm: Some(0),
            scheme: "variance_stabilizing".to_owned(),
            prior_mean: 0.5,
            prior_strength: 1.0,
            size_weighted: false,
        };
        let direct = session.measured_interval(&weighting, "weighting", &ctx()).unwrap_err();
        let compiled = Study::compile_adaptive_interval(&session, weighting, "weighting").unwrap();
        assert_eq!(compiled.execute(&ctx()).unwrap_err(), direct);
    }

    #[test]
    fn the_adaptive_program_identity_binds_the_request_and_the_contract() {
        let first = session(thompson_trial([0.3, 0.5], 200, &[100, 200], 5));
        let second = session(thompson_trial([0.3, 0.5], 200, &[100, 200], 6));
        let identity = |session: &AdaptiveInferenceSession, request: AdaptiveRequest, id: &str| {
            Study::compile_adaptive_interval(session, request, id).unwrap().lowered().identity()
        };
        assert_eq!(identity(&first, posterior(), "a"), identity(&first, posterior(), "a"));
        assert_ne!(identity(&first, posterior(), "a"), identity(&first, posterior(), "b"));
        assert_ne!(identity(&first, posterior(), "a"), identity(&second, posterior(), "a"));
    }

    #[test]
    fn the_compiled_thompson_interval_is_the_recounted_beta_posterior() {
        let declaration = thompson_trial([0.3, 0.5], 200, &[100, 200], 5);
        // Successes and failures of the second arm over every unit available at the last look.
        let (mut successes, mut failures) = (0_u32, 0_u32);
        for (arm, outcome) in declaration.assigned_arm.iter().zip(&declaration.outcomes) {
            if *arm == 1 {
                if outcome[0] > 0.5 {
                    successes += 1;
                } else {
                    failures += 1;
                }
            }
        }
        let (a, b) = (1 + successes, 1 + failures);
        let session = session(declaration);
        let run = Study::compile_adaptive_interval(&session, posterior(), "thompson")
            .unwrap()
            .execute(&ctx())
            .expect("reported");
        let scalar = &run.measured.report().scalars[0];
        assert_eq!(scalar.name, "treatment_success_probability");
        let mean = f64::from(a) / f64::from(a + b);
        assert!((scalar.point - mean).abs() < 1e-12, "point {} vs {mean}", scalar.point);
        let tail = (1.0 - scalar.level) / 2.0;
        assert!((scalar.level - 0.95).abs() < 1e-12);
        let (lower, upper) = (beta_quantile(a, b, tail), beta_quantile(a, b, 1.0 - tail));
        assert!((scalar.lower - lower).abs() < 1e-7, "lower {} vs {lower}", scalar.lower);
        assert!((scalar.upper - upper).abs() < 1e-7, "upper {} vs {upper}", scalar.upper);
    }
}

/// A categorical choice study: four of six respondents choose the first of three options.
mod choice {
    use super::{Study, assert_twin, beta_quantile, ctx};
    use antecedent::analysis::choice_inference::{
        ChoiceDesignDeclaration, ChoiceInferenceSession, ChoiceLook, ChoiceRequest,
        ChoiceRespondent,
    };
    use antecedent_contract::adaptive_design::{
        AssignmentFamily, MonitoringStrategy, OutcomeStructure, ProcedureCell,
        StoppingRuleDeclaration, StoppingRuleKind, StoppingSpec,
    };

    fn study() -> ChoiceDesignDeclaration {
        let picks = [0, 0, 0, 0, 1, 2];
        ChoiceDesignDeclaration {
            label: "pricing".into(),
            cell: ProcedureCell {
                assignment: AssignmentFamily::FixedRandomization,
                outcome: OutcomeStructure::CategoricalChoice,
                monitoring: MonitoringStrategy::FinalAnalysisOnly,
                stopping: StoppingRuleKind::NoStopping,
            },
            option_labels: ["a", "b", "c"].map(String::from).to_vec(),
            meaning: "chosen in the pricing task".into(),
            population: "customers".into(),
            respondents: picks
                .iter()
                .enumerate()
                .map(|(i, p)| {
                    let mut counts = vec![0, 0, 0];
                    counts[*p] = 1;
                    ChoiceRespondent { id: format!("r{i}"), counts }
                })
                .collect(),
            initial_prior: vec![1.0; 3],
            retained_sources: vec![],
            looks: Some(vec![ChoiceLook { respondents: 6 }]),
            stopping_rule: Some(StoppingRuleDeclaration {
                kind: StoppingRuleKind::NoStopping,
                spec: StoppingSpec::NoStopping,
                ended_by_rule: false,
            }),
        }
    }

    fn session() -> ChoiceInferenceSession {
        ChoiceInferenceSession::prepare(study()).expect("complete study")
    }

    #[test]
    fn the_compiled_choice_interval_equals_the_direct_interval() {
        let session = session();
        let request = ChoiceRequest::Posterior { look: 0 };
        let direct = session.measured_interval(&request, "choice", &ctx()).expect("reported");
        let run = Study::compile_choice_interval(&session, request, "choice")
            .unwrap()
            .execute(&ctx())
            .expect("reported");
        assert_twin(&direct, &run);
        assert_eq!(run.calibration_route, "choice_interval");
        // A look the study does not declare refuses identically.
        let missing = ChoiceRequest::Posterior { look: 3 };
        let direct = session.measured_interval(&missing, "missing", &ctx()).unwrap_err();
        let compiled = Study::compile_choice_interval(&session, missing, "missing").unwrap();
        assert_eq!(compiled.execute(&ctx()).unwrap_err(), direct);
    }

    #[test]
    fn the_compiled_choice_interval_is_the_beta_marginal_of_the_dirichlet_posterior() {
        // Uniform Dirichlet(1, 1, 1) and counts (4, 1, 1): the posterior is Dirichlet(5, 2, 2),
        // whose first share is Beta(5, 4) with mean 5/9.
        let session = session();
        let run =
            Study::compile_choice_interval(&session, ChoiceRequest::Posterior { look: 0 }, "c")
                .unwrap()
                .execute(&ctx())
                .expect("reported");
        let scalar = &run.measured.report().scalars[0];
        assert!((scalar.point - 5.0 / 9.0).abs() < 1e-12, "point {}", scalar.point);
        let tail = (1.0 - scalar.level) / 2.0;
        let (lower, upper) = (beta_quantile(5, 4, tail), beta_quantile(5, 4, 1.0 - tail));
        assert!((scalar.lower - lower).abs() < 1e-7, "lower {} vs {lower}", scalar.lower);
        assert!((scalar.upper - upper).abs() < 1e-7, "upper {} vs {upper}", scalar.upper);
    }
}

/// The exact-count dynamic SCM panel of the estimator tests.
mod temporal {
    use super::{Study, assert_twin, ctx};
    use antecedent::analysis::compiled_temporal_interval::TemporalDependentIntervalRequest;
    use antecedent::analysis::{IntervalEstimand, temporal_dependent_interval};
    use antecedent_core::ExecutionContext;
    use antecedent_estimate::temporal_dependent_interval::{
        DependentIntervalConfig, SequenceHistory, UnitHistories,
    };
    use antecedent_estimate::temporal_initial_state::{
        InitialStateLaw, InitialStatePopulation, InitialStateSpec,
    };

    /// `P(l = 1 | s0, a1)` in tenths, indexed `[s0][a1]`.
    const P_L1: [[u64; 2]; 2] = [[3, 7], [2, 8]];
    /// Outcome mean in tenths, indexed `[s0][a1][l][a2]`.
    const M10: [[[[i64; 2]; 2]; 2]; 2] =
        [[[[3, 4], [4, 6]], [[3, 5], [5, 7]]], [[[4, 3], [7, 6]], [[6, 3], [5, 7]]]];
    const SEQUENCE: [u32; 2] = [0, 0];
    const TARGET: [(u32, f64); 2] = [(0, 0.3), (1, 0.7)];
    const SNAPSHOT: &str = "dynamic-scm-panel";

    fn unit_rows(unit: usize) -> Vec<(u32, u32, u32, u32, f64)> {
        let shift: i64 = if unit.is_multiple_of(2) { 2 } else { -2 };
        let mut rows = Vec::new();
        for s0 in 0..2_usize {
            for a1 in 0..2_usize {
                let n_one = 20 * P_L1[s0][a1];
                for l in 0..2_usize {
                    let n_l = if l == 1 { n_one } else { 200 - n_one };
                    for a2 in 0..2_usize {
                        let n_cell = n_l / 2;
                        let ones = n_cell / 10 * (M10[s0][a1][l][a2] + shift) as u64;
                        for k in 0..n_cell {
                            let y = if k < ones { 1.0 } else { 0.0 };
                            rows.push((s0 as u32, a1 as u32, l as u32, a2 as u32, y));
                        }
                    }
                }
            }
        }
        rows
    }

    fn units(count: usize) -> Vec<UnitHistories> {
        (0..count)
            .map(|unit| UnitHistories {
                unit_id: unit as u64,
                histories: unit_rows(unit)
                    .into_iter()
                    .enumerate()
                    .map(|(t, r)| SequenceHistory::two_step(t as u64, r.0, r.1, r.2, r.3, r.4))
                    .collect(),
            })
            .collect()
    }

    fn estimand() -> IntervalEstimand {
        let law =
            InitialStateLaw::new(InitialStatePopulation::Target, "target-state", TARGET.to_vec())
                .unwrap();
        IntervalEstimand::Marginalized(InitialStateSpec::Law(law))
    }

    fn config() -> DependentIntervalConfig {
        DependentIntervalConfig { replicates: 200, seed: 7, ..DependentIntervalConfig::default() }
    }

    fn request() -> TemporalDependentIntervalRequest {
        TemporalDependentIntervalRequest::new(
            SNAPSHOT,
            Some(units(30)),
            SEQUENCE,
            estimand(),
            config(),
        )
    }

    #[test]
    fn the_compiled_temporal_interval_equals_the_direct_interval() {
        let direct =
            temporal_dependent_interval(SNAPSHOT, Some(units(30)), SEQUENCE, estimand(), &config())
                .expect("reported");
        let compiled = Study::compile_temporal_dependent_interval(request()).unwrap();
        // The direct entry runs under the production context of the design's seed; the
        // compiled route under the caller's. The stream is seeded from the design.
        for context in [ExecutionContext::production_default(config().seed), ctx()] {
            let run = compiled.execute(&context).expect("reported");
            assert_twin(&direct, &run);
            assert_eq!(run.calibration_route, "temporal_interval");
        }
        // Too few units refuse identically.
        let short = TemporalDependentIntervalRequest::new(
            SNAPSHOT,
            Some(units(4)),
            SEQUENCE,
            estimand(),
            config(),
        );
        let direct =
            temporal_dependent_interval(SNAPSHOT, Some(units(4)), SEQUENCE, estimand(), &config())
                .unwrap_err();
        let refused =
            Study::compile_temporal_dependent_interval(short).unwrap().execute(&ctx()).unwrap_err();
        assert_eq!(format!("{refused:?}"), format!("{direct:?}"));
    }

    #[test]
    fn the_temporal_program_identity_binds_the_units_and_the_design() {
        let identity = |request: TemporalDependentIntervalRequest| {
            Study::compile_temporal_dependent_interval(request).unwrap().lowered().identity()
        };
        assert_eq!(identity(request()), identity(request()));
        let reseeded = TemporalDependentIntervalRequest::new(
            SNAPSHOT,
            Some(units(30)),
            SEQUENCE,
            estimand(),
            DependentIntervalConfig { seed: 8, ..config() },
        );
        assert_ne!(identity(request()), identity(reseeded));
        let mut changed = units(30);
        changed[0].histories[0].y = 1.0 - changed[0].histories[0].y;
        let edited = TemporalDependentIntervalRequest::new(
            SNAPSHOT,
            Some(changed),
            SEQUENCE,
            estimand(),
            config(),
        );
        assert_ne!(identity(request()), identity(edited));
    }

    #[test]
    fn the_compiled_temporal_interval_reports_the_enumerated_marginalized_response() {
        let run = Study::compile_temporal_dependent_interval(request())
            .unwrap()
            .execute(&ctx())
            .expect("reported");
        let scalar = run
            .measured
            .report()
            .scalars
            .iter()
            .find(|scalar| scalar.name == "response")
            .expect("the response scalar");
        // 0.3 * 0.33 + 0.7 * 0.46, enumerated from the SCM tables.
        assert!((scalar.point - 0.421).abs() < 1e-12, "point {}", scalar.point);
        assert!(scalar.lower < 0.421 && 0.421 < scalar.upper);
    }
}
