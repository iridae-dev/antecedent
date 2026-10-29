//! Acceptance evidence for the adaptive-policy search.
//!
//! The response simulators below are independent of the code under test: they hold their own
//! random generator and dynamics, and every reference value (the treatment-threshold optimum,
//! the Riccati linear-feedback optimum and the best open-loop schedule) is a closed form
//! derived by hand, never read back from the search. `conformance/adaptive_policy_search.py`
//! recomputes the same references in a standalone script.

use antecedent_core::{ExecutionContext, VariableId};
use antecedent_design::adaptive_policy::{AdaptivePolicyFamily, PolicyFeature, PolicyRule};
use antecedent_design::adaptive_policy_search::{
    AdaptivePolicySearchError, ChanceBound, DrawSplit, HeldOutDesign, HeldOutRequirement,
    PolicyConstraints, PolicyObjective, PolicyPath, PolicyRolloutSource, PolicySearchOptions,
    PolicyStanding, PreparedAdaptivePolicySearch, ResponseStanding, SearchStop, StandingRequest,
};

const STATE: VariableId = VariableId::from_raw(0);

struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        let mut s = Self(seed ^ 0x9E37_79B9_7F4A_7C15);
        for _ in 0..4 {
            s.next();
        }
        s
    }
    fn next(&mut self) -> u64 {
        self.0 =
            self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        let mut z = self.0;
        z ^= z >> 33;
        z = z.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
        z ^ (z >> 33)
    }
    fn unit(&mut self) -> f64 {
        ((self.next() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f64 {
        (-2.0 * self.unit().ln()).sqrt() * (2.0 * std::f64::consts::PI * self.unit()).cos()
    }
}

#[derive(Clone, Copy)]
enum Dynamics {
    /// z_{t+1} = rho z_t + sigma e; utility sum u_t (z_t - k); cost unit per treatment.
    Treatment { rho: f64, sigma: f64, k: f64 },
    /// x_{t+1} = a x_t + u_t + w; utility -(sum q x^2 + r u^2) - qf x_T^2.
    Lq { a: f64, q: f64, r: f64, qf: f64, sigma: f64, x0: f64 },
}

struct Sim {
    dynamics: Dynamics,
    horizon: usize,
    n_opt: usize,
    n_held: usize,
    identified: bool,
    held_out: HeldOutDesign,
    observable: Option<Vec<VariableId>>,
}

impl Sim {
    fn treatment() -> Self {
        Self {
            dynamics: Dynamics::Treatment { rho: 0.6, sigma: 0.8, k: 0.5 },
            horizon: 10,
            n_opt: 3000,
            n_held: 4000,
            identified: false,
            held_out: HeldOutDesign::IndependentDraws,
            observable: None,
        }
    }
    fn lq() -> Self {
        Self {
            dynamics: Dynamics::Lq {
                a: 0.9,
                q: 1.0,
                r: 0.5,
                qf: lq_p(0.9, 1.0, 0.5),
                sigma: 1.0,
                x0: 2.0,
            },
            horizon: 12,
            n_opt: 3000,
            n_held: 4000,
            identified: false,
            held_out: HeldOutDesign::IndependentDraws,
            observable: None,
        }
    }
}

impl PolicyRolloutSource for Sim {
    fn identity(&self) -> String {
        "independent-simulator".into()
    }
    fn n_draws(&self, split: DrawSplit) -> usize {
        match split {
            DrawSplit::Optimization => self.n_opt,
            DrawSplit::HeldOut => self.n_held,
        }
    }
    fn horizon(&self) -> usize {
        self.horizon
    }
    fn standing(&self) -> ResponseStanding {
        ResponseStanding {
            identified: self.identified,
            basis: if self.identified {
                "randomised sequential assignment with sequential exchangeability".into()
            } else {
                "observational fit; sequential exchangeability not established".into()
            },
        }
    }
    fn heldout_design(&self) -> HeldOutDesign {
        self.held_out.clone()
    }
    fn observable_variables(&self) -> Option<Vec<VariableId>> {
        self.observable.clone()
    }
    fn rollout(
        &self,
        family: &AdaptivePolicyFamily,
        theta: &[f64],
        split: DrawSplit,
        _ctx: &ExecutionContext,
    ) -> Result<Vec<PolicyPath>, AdaptivePolicySearchError> {
        let (n, base) = match split {
            DrawSplit::Optimization => (self.n_opt, 1_000_003_u64),
            DrawSplit::HeldOut => (self.n_held, 7_777_777_u64),
        };
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let mut rng = Lcg::new(base.wrapping_mul(1_000_003).wrapping_add(i as u64));
            let mut utility = 0.0;
            let (mut actions, mut costs, mut exposures) = (vec![], vec![], vec![]);
            match self.dynamics {
                Dynamics::Treatment { rho, sigma, k } => {
                    let stationary = sigma / (1.0 - rho * rho).sqrt();
                    let mut z = stationary * rng.normal();
                    for t in 0..self.horizon {
                        let u = family
                            .action(theta, t, &[z])
                            .map_err(|e| AdaptivePolicySearchError::Source(e.to_string()))?;
                        utility += u * (z - k);
                        actions.push(u);
                        costs.push(u);
                        exposures.push(u);
                        z = rho * z + sigma * rng.normal();
                    }
                }
                Dynamics::Lq { a, q, r, qf, sigma, x0 } => {
                    let mut x = x0;
                    for t in 0..self.horizon {
                        let u = family
                            .action(theta, t, &[x])
                            .map_err(|e| AdaptivePolicySearchError::Source(e.to_string()))?;
                        utility -= q * x * x + r * u * u;
                        actions.push(u);
                        costs.push(u.abs());
                        exposures.push(u.abs());
                        x = a * x + u + sigma * rng.normal();
                    }
                    utility -= qf * x * x;
                }
            }
            out.push(PolicyPath { utility, actions, step_costs: costs, step_exposures: exposures });
        }
        Ok(out)
    }
}

// ---- hand-derived references -------------------------------------------------------------

fn norm_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt()
}
fn norm_cdf(x: f64) -> f64 {
    // Abramowitz and Stegun 7.1.26 via erf; absolute error below 1.5e-7.
    let t = 1.0 / (1.0 + 0.327_591_1 * (x.abs() / 2.0_f64.sqrt()));
    let poly = t
        * (0.254_829_592
            + t * (-0.284_496_736
                + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    let erf = 1.0 - poly * (-(x * x) / 2.0).exp();
    0.5 * (1.0 + if x >= 0.0 { erf } else { -erf })
}
/// Per-step value of the optimal treat-if-above-k rule on a stationary N(0, s^2) state.
fn treatment_step_value(s: f64, k: f64) -> f64 {
    s * norm_pdf(k / s) - k * (1.0 - norm_cdf(k / s))
}
/// Steady-state Riccati solution of P = q + a^2 P - (a P)^2 / (r + P) with b = 1.
fn lq_p(a: f64, q: f64, r: f64) -> f64 {
    let mut p = q;
    for _ in 0..10_000 {
        p = q + a * a * p - (a * p) * (a * p) / (r + p);
    }
    p
}

fn threshold_family(id: &str) -> AdaptivePolicyFamily {
    AdaptivePolicyFamily::new(
        id,
        vec![PolicyFeature::Current(STATE)],
        &[(-4.0, 4.0)],
        PolicyRule::Threshold { feature: 0 },
        (0.0, 1.0),
        (-1.0, 1.0),
    )
    .unwrap()
}
fn schedule_family(steps: usize, lo: f64, hi: f64) -> AdaptivePolicyFamily {
    AdaptivePolicyFamily::new(
        "schedule",
        vec![PolicyFeature::Current(STATE)],
        &[(-4.0, 4.0)],
        PolicyRule::Schedule { steps },
        (lo, hi),
        (-1.0, 1.0),
    )
    .unwrap()
}
fn feedback_family() -> AdaptivePolicyFamily {
    AdaptivePolicyFamily::new(
        "feedback",
        vec![PolicyFeature::Current(STATE)],
        &[(-10.0, 10.0)],
        PolicyRule::LinearFeedback { features: vec![0] },
        (-6.0, 6.0),
        (-2.0, 2.0),
    )
    .unwrap()
}

fn options(seed: u64) -> PolicySearchOptions {
    PolicySearchOptions { seed, max_evaluations: 1500, ..PolicySearchOptions::default() }
}

fn run(
    families: Vec<AdaptivePolicyFamily>,
    constraints: PolicyConstraints,
    sim: &Sim,
    opts: PolicySearchOptions,
) -> Result<
    antecedent_design::adaptive_policy_search::AdaptivePolicySearchResult,
    AdaptivePolicySearchError,
> {
    let ctx = ExecutionContext::for_tests(5);
    PreparedAdaptivePolicySearch::compile(
        families,
        Some(0),
        constraints,
        PolicyObjective::ExpectedUtility,
        opts,
    )?
    .execute(sim, &ctx)
}

#[test]
fn state_dependent_threshold_beats_the_best_fixed_schedule_at_the_closed_form_optimum() {
    let sim = Sim::treatment();
    let result = run(
        vec![schedule_family(10, 0.0, 1.0), threshold_family("threshold")],
        PolicyConstraints::default(),
        &sim,
        options(11),
    )
    .unwrap();
    // The state is stationary N(0, 1) (sigma / sqrt(1 - rho^2) = 1), k = 0.5.
    let optimum = 10.0 * treatment_step_value(1.0, 0.5);
    assert!((optimum - 1.978).abs() < 0.01, "reference value {optimum}");
    let threshold = &result.families[1];
    let held = threshold.held_out.as_ref().unwrap();
    assert!(
        (held.mean_utility - optimum).abs() < 4.0 * held.standard_error + 0.08,
        "held-out {} vs closed form {optimum}",
        held.mean_utility
    );
    assert!(held.mean_utility <= optimum + 4.0 * held.standard_error);
    // Any fixed schedule earns -k per treated step on average, so the best schedule is "never".
    let schedule = result.families[0].held_out.as_ref().unwrap();
    assert!(schedule.mean_utility.abs() < 4.0 * schedule.standard_error + 0.02);
    assert!(result.families[0].theta.iter().all(|u| *u < 0.15));
    assert_eq!(result.selected, Some(1));
    let contrast = result
        .contrasts
        .iter()
        .find(|c| c.split == DrawSplit::HeldOut)
        .expect("held-out paired contrast");
    assert_eq!(contrast.baseline, "schedule");
    assert!((contrast.mean_difference - (held.mean_utility - schedule.mean_utility)).abs() < 1e-9);
    assert!(contrast.mean_difference > 20.0 * contrast.standard_error);
    assert!(contrast.mean_difference > 0.85 * optimum);
    assert!((threshold.theta[0] - 0.5).abs() < 0.4, "threshold near k");
}

#[test]
fn linear_feedback_reaches_the_riccati_optimum_and_beats_the_open_loop_optimum() {
    let sim = Sim::lq();
    let (a, q, r, sigma, x0, t) = (0.9_f64, 1.0_f64, 0.5_f64, 1.0_f64, 2.0_f64, 12.0_f64);
    let p = lq_p(a, q, r);
    // With terminal weight P the finite-horizon Riccati gain is stationary, so the closed-loop
    // expected cost is P x0^2 + T P sigma^2 and the gain is K = a P / (r + P).
    let closed_loop = -(p * x0 * x0 + t * p * sigma * sigma);
    let gain = a * p / (r + p);
    // Open loop: deterministic optimum P x0^2 plus the input-independent state variance cost.
    let mut noise_cost = 0.0;
    for step in 1..=12 {
        let variance: f64 = (0..step).map(|j| a.powi(2 * j)).sum::<f64>() * sigma * sigma;
        noise_cost += if step < 12 { q * variance } else { p * variance };
    }
    let open_loop = -(p * x0 * x0 + noise_cost);
    assert!(closed_loop - open_loop > 10.0, "the reference gap is material");

    let result = run(
        vec![schedule_family(12, -6.0, 6.0), feedback_family()],
        PolicyConstraints::default(),
        &sim,
        options(3),
    )
    .unwrap();
    let feedback = &result.families[1];
    let held = feedback.held_out.as_ref().unwrap();
    assert!(
        (held.mean_utility - closed_loop).abs() < 5.0 * held.standard_error + 0.05,
        "held-out {} vs Riccati {closed_loop}",
        held.mean_utility
    );
    assert!((feedback.theta[1] + gain).abs() < 0.12, "slope {} vs -K {}", feedback.theta[1], -gain);
    let schedule = result.families[0].held_out.as_ref().unwrap();
    assert!(
        (schedule.mean_utility - open_loop).abs() < 5.0 * schedule.standard_error + 0.3,
        "schedule {} vs open-loop optimum {open_loop}",
        schedule.mean_utility
    );
    assert!(held.mean_utility > schedule.mean_utility + 8.0);
}

#[test]
fn hard_and_chance_exposure_constraints_bind_and_relax_monotonically() {
    let sim = Sim::treatment();
    let free =
        run(vec![threshold_family("threshold")], PolicyConstraints::default(), &sim, options(2))
            .unwrap();
    let hard = PolicyConstraints {
        cumulative_exposure: Some(ChanceBound { bound: 3.0, max_violation_probability: 0.0 }),
        ..PolicyConstraints::default()
    };
    let chance = PolicyConstraints {
        cumulative_exposure: Some(ChanceBound { bound: 3.0, max_violation_probability: 0.2 }),
        ..PolicyConstraints::default()
    };
    let hard_result = run(vec![threshold_family("threshold")], hard, &sim, options(2)).unwrap();
    let chance_result = run(vec![threshold_family("threshold")], chance, &sim, options(2)).unwrap();
    let h = &hard_result.families[0].in_sample;
    let report = &h.constraints[0];
    assert!(h.feasible && report.violations == 0 && report.worst_observed <= 3.0 + 1e-12);
    assert!(!report.upper_bound_satisfied, "zero tolerance is never certified by a finite sample");
    let c = &chance_result.families[0].in_sample;
    let creport = &c.constraints[0];
    assert!(c.feasible && creport.violation_rate <= 0.2);
    assert!(creport.violation_rate_upper >= creport.violation_rate);
    let f = free.families[0].in_sample.mean_utility;
    assert!(hard_result.families[0].in_sample.mean_utility < f - 0.1, "the hard bound costs value");
    assert!(c.mean_utility >= h.mean_utility - 0.1, "a chance relaxation cannot lose value");
    assert!(c.mean_utility <= f + 0.1);
    assert!(hard_result.selected.is_some());
}

#[test]
fn per_step_action_cost_and_infeasible_constraints_are_enforced_on_paths() {
    let sim = Sim::treatment();
    // The family can act up to 1.0; the realised-action constraint caps it at 0.5.
    let capped =
        PolicyConstraints { per_step_action: Some((0.0, 0.5)), ..PolicyConstraints::default() };
    let result = run(vec![threshold_family("threshold")], capped, &sim, options(4)).unwrap();
    let e = &result.families[0].in_sample;
    assert!(e.feasible && e.constraints[0].violations == 0);
    assert!(result.families[0].theta[2] <= 0.5 + 1e-12);

    let cost = PolicyConstraints {
        cumulative_cost: Some(ChanceBound { bound: -1.0, max_violation_probability: 0.0 }),
        ..PolicyConstraints::default()
    };
    let infeasible = run(vec![threshold_family("threshold")], cost, &sim, options(4)).unwrap();
    assert_eq!(infeasible.selected, None);
    assert!(!infeasible.families[0].in_sample.feasible);
    assert!(infeasible.flags.iter().any(|f| f.contains("satisfying every constraint")));
}

#[test]
fn standing_is_fitted_model_unless_requested_and_supported() {
    let mut sim = Sim::treatment();
    let mut opts = options(1);
    opts.max_evaluations = 200;
    let label = |sim: &Sim, standing| {
        let mut o = opts.clone();
        o.standing = standing;
        run(vec![threshold_family("t")], PolicyConstraints::default(), sim, o)
    };
    let r = label(&sim, StandingRequest::FittedModel).unwrap();
    assert_eq!(r.standing.label(), "fitted_model_optimization");
    let refused = label(&sim, StandingRequest::IdentifiedPolicyEffect).unwrap_err();
    assert!(matches!(refused, AdaptivePolicySearchError::IdentificationUnsupported(_)));
    assert_eq!(refused.reason_code(), Some("adaptive_policy_identification_unsupported"));
    sim.identified = true;
    let r = label(&sim, StandingRequest::IdentifiedPolicyEffect).unwrap();
    assert!(matches!(r.standing, PolicyStanding::IdentifiedPolicyEffect { .. }));
    let r = label(&sim, StandingRequest::FittedModel).unwrap();
    assert_eq!(r.standing.label(), "fitted_model_optimization", "identified but not requested");
}

#[test]
fn held_out_evaluation_is_required_or_flagged_and_reports_selection_optimism() {
    let mut sim = Sim::treatment();
    let mut opts = options(6);
    opts.max_evaluations = 300;
    let with =
        run(vec![threshold_family("t")], PolicyConstraints::default(), &sim, opts.clone()).unwrap();
    assert!(with.selection_optimism.is_some());
    sim.held_out = HeldOutDesign::Unavailable("one fitted cohort; no separated units".into());
    sim.n_held = 0;
    let flagged =
        run(vec![threshold_family("t")], PolicyConstraints::default(), &sim, opts.clone()).unwrap();
    assert!(flagged.families[0].held_out.is_none() && flagged.selection_optimism.is_none());
    assert!(flagged.flags.iter().any(|f| f.contains("held-out evaluation unavailable")));
    opts.held_out = HeldOutRequirement::Required;
    let refused =
        run(vec![threshold_family("t")], PolicyConstraints::default(), &sim, opts).unwrap_err();
    assert!(matches!(refused, AdaptivePolicySearchError::HeldOutUnsupported(_)));
}

#[test]
fn refusals_are_typed_for_unobserved_features_invalid_problems_budget_and_cancellation() {
    let mut sim = Sim::treatment();
    sim.observable = Some(vec![VariableId::from_raw(9)]);
    let err = run(vec![threshold_family("t")], PolicyConstraints::default(), &sim, options(1))
        .unwrap_err();
    assert!(matches!(err, AdaptivePolicySearchError::UnobservedFeature { variable: 0 }));
    // An open-loop schedule reads nothing, so it needs no observed variable.
    assert!(run(
        vec![schedule_family(10, 0.0, 1.0)],
        PolicyConstraints::default(),
        &sim,
        PolicySearchOptions { max_evaluations: 60, ..options(1) }
    )
    .is_ok());

    let ok = Sim::treatment();
    let compile = |families: Vec<AdaptivePolicyFamily>, o: PolicySearchOptions| {
        PreparedAdaptivePolicySearch::compile(
            families,
            None,
            PolicyConstraints::default(),
            PolicyObjective::ExpectedUtility,
            o,
        )
    };
    // Each invalid problem is the operational `Invalid` refusal (no scientific reason code).
    let invalid = |r: Result<PreparedAdaptivePolicySearch, AdaptivePolicySearchError>| match r {
        Err(AdaptivePolicySearchError::Invalid(message)) => message,
        Err(other) => panic!("expected Invalid, got {other:?}"),
        Ok(_) => panic!("expected Invalid, compiled"),
    };
    assert!(!invalid(compile(vec![], options(1))).is_empty());
    let duplicate = invalid(compile(
        vec![threshold_family("a"), threshold_family("a")],
        options(1),
    ));
    assert!(!duplicate.is_empty(), "{duplicate}");
    let elite = invalid(compile(
        vec![threshold_family("a")],
        PolicySearchOptions { elite_fraction: 0.9, ..options(1) },
    ));
    assert_eq!(elite, "invalid bounded search policy");
    let _ = invalid(PreparedAdaptivePolicySearch::compile(
        vec![threshold_family("a")],
        Some(3),
        PolicyConstraints::default(),
        PolicyObjective::ExpectedUtility,
        options(1),
    ));
    let alpha = invalid(PreparedAdaptivePolicySearch::compile(
        vec![threshold_family("a")],
        None,
        PolicyConstraints::default(),
        PolicyObjective::LowerTailMean { alpha: 0.0 },
        options(1),
    ));
    assert!(!alpha.is_empty(), "{alpha}");

    let budget = run(
        vec![threshold_family("t")],
        PolicyConstraints::default(),
        &ok,
        PolicySearchOptions { max_evaluations: 10, ..options(1) },
    )
    .unwrap();
    assert!(!budget.complete && budget.families[0].stop == SearchStop::EvaluationBudget);
    assert!(budget.flags.iter().any(|f| f.contains("incomplete")));

    let ctx = ExecutionContext::for_tests(1);
    ctx.cancellation.cancel();
    let prepared = compile(vec![threshold_family("t")], options(1)).unwrap();
    assert!(matches!(prepared.execute(&ok, &ctx), Err(AdaptivePolicySearchError::Cancelled)));

    let mut tight = ExecutionContext::for_tests(1);
    tight.set_memory_limit(Some(1_000));
    assert!(matches!(prepared.execute(&ok, &tight), Err(AdaptivePolicySearchError::Resource(_))));
}

#[test]
fn a_fixed_seed_replays_exactly_and_a_new_seed_changes_the_record() {
    let sim = Sim::treatment();
    let o = PolicySearchOptions { max_evaluations: 150, ..options(21) };
    let make = |o: PolicySearchOptions| {
        PreparedAdaptivePolicySearch::compile(
            vec![schedule_family(10, 0.0, 1.0), threshold_family("t")],
            Some(0),
            PolicyConstraints::default(),
            PolicyObjective::ExpectedUtility,
            o,
        )
        .unwrap()
    };
    let ctx = ExecutionContext::for_tests(5);
    let first = make(o.clone()).execute(&sim, &ctx).unwrap();
    assert!(make(o.clone()).verify_replay(&first, &sim, &ctx).unwrap());
    let other = make(PolicySearchOptions { seed: 22, ..o }).execute(&sim, &ctx).unwrap();
    assert_ne!(first.record_identity, other.record_identity);
    assert!(first.admitted_bytes() > 0);
}

#[test]
fn lower_tail_objective_is_no_larger_than_the_mean() {
    let sim = Sim::treatment();
    let ctx = ExecutionContext::for_tests(5);
    let run_with = |objective| {
        PreparedAdaptivePolicySearch::compile(
            vec![threshold_family("t")],
            None,
            PolicyConstraints::default(),
            objective,
            PolicySearchOptions { max_evaluations: 200, ..options(8) },
        )
        .unwrap()
        .execute(&sim, &ctx)
        .unwrap()
    };
    let mean = run_with(PolicyObjective::ExpectedUtility);
    let tail = run_with(PolicyObjective::LowerTailMean { alpha: 0.1 });
    assert!(tail.families[0].in_sample.objective < mean.families[0].in_sample.objective);
}

#[test]
fn the_in_test_closed_forms_match_the_conformance_document() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../conformance/design/adaptive_policy_search/expected.json");
    let doc: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let get = |a: &str, b: &str| doc[a][b].as_f64().unwrap();
    let treatment = 10.0 * treatment_step_value(1.0, 0.5);
    assert!((treatment - get("treatment", "adaptive_value")).abs() < 1e-5, "A&S cdf error bound");
    assert_eq!(get("treatment", "best_fixed_schedule_value"), 0.0);
    let p = lq_p(0.9, 1.0, 0.5);
    assert!((p - get("lq", "P")).abs() < 1e-12);
    assert!((-(p * 4.0 + 12.0 * p) - get("lq", "closed_loop_value")).abs() < 1e-9);
    assert!((0.9 * p / (0.5 + p) - get("lq", "gain")).abs() < 1e-12);
}
