//! B4 nonlinear continuous-mediator mediation: exact-SCM closed-form truth, quadrature error
//! check, confounding and overlap refusals, the total = direct + indirect identity, and a
//! labelled regression cross-check against the linear product of coefficients. Calibration is
//! `unmeasured` and nothing here measures coverage.
//!
//! SCM: `M = alpha*A + gamma*X + e_M`, `e_M ~ N(0, sigma2)`;
//! `Y = b0 + b1*A + c1*M + c2*M^2 + delta*A*M + bx*X` (no outcome noise, so the outcome fit is
//! exact). `X` is uniform on `{-1, 0, 1, 2, 3}`: `E[X] = 1`, `E[X^2] = 3`.
//!
//! Closed form via `E[M | a', X] = alpha*a' + gamma*X` and
//! `E[M^2 | a', X] = (alpha*a' + gamma*X)^2 + sigma2`, averaged over `X`:
//!   `m1(a') = alpha*a' + gamma*E[X]`
//!   `m2(a') = (alpha*a')^2 + 2*alpha*a'*gamma*E[X] + gamma^2*E[X^2] + sigma2`
//!   `theta(a, a') = b0 + b1*a + bx*E[X] + (c1 + delta*a)*m1(a') + c2*m2(a')`
//!   `NDE = theta(1,0) - theta(0,0) = b1 + delta*gamma*E[X]`
//!   `NIE = theta(1,1) - theta(1,0) = (c1 + delta)*alpha + c2*(alpha^2 + 2*alpha*gamma*E[X])`
//!   `TE  = theta(1,1) - theta(0,0)`.
//! A cubic term `c3*M^3` (zero in the base SCM) adds `c3*m3(a')` to `theta`, with
//! `E[M^3 | a', X] = mu^3 + 3*mu*sigma2`, `mu = alpha*a' + gamma*X`, hence
//! `m3(a') = E_X[mu^3] + 3*sigma2*E_X[mu]` and `E_X[mu^3]` expands with `E[X^3] = 7`.
//! The homoscedastic `sigma2` cancels from every contrast for a quadratic `g`, so only the cubic
//! variant exercises the quadrature node count.
//!
//! The dataset is a deterministic crossed grid (A in {0,1}) x (X in the five values) x
//! (`e_M` in {-b, -a, a, b}); every cell holds the four symmetric residuals, so the residuals are
//! exactly orthogonal to `(1, A, X)` and the OLS mediator fit recovers `alpha`, `gamma` exactly.
//! The residual levels are scaled so that the degrees-of-freedom corrected residual variance
//! `RSS / (n - 3)` equals `sigma2` exactly (`mean(e^2) = sigma2 * (n - 3) / n`).

use antecedent_estimate::EstimationError;
use antecedent_estimate::nonlinear_mediation::{
    NonlinearMediationConfig, NonlinearMediationEstimand, NonlinearMediationEstimate,
    NonlinearMediationInput, NonlinearMediationPremises, estimate_nonlinear_mediation,
};

const XS: [f64; 5] = [-1.0, 0.0, 1.0, 2.0, 3.0];
const EX: f64 = 1.0;
const EX2: f64 = 3.0;
const EX3: f64 = 7.0;

#[derive(Clone, Copy)]
struct Scm {
    alpha: f64,
    gamma: f64,
    sigma2: f64,
    b0: f64,
    b1: f64,
    c1: f64,
    c2: f64,
    c3: f64,
    delta: f64,
    bx: f64,
}

fn base_scm() -> Scm {
    Scm {
        alpha: 0.5,
        gamma: 0.5,
        sigma2: 0.64,
        b0: 1.0,
        b1: 0.8,
        c1: 0.6,
        c2: 0.4,
        c3: 0.0,
        delta: 0.3,
        bx: 0.7,
    }
}

struct Grid {
    a: Vec<f64>,
    m: Vec<f64>,
    y: Vec<f64>,
    x: Vec<f64>,
}

fn grid(s: &Scm) -> Grid {
    let rows = 2.0 * 5.0 * 4.0;
    let v = s.sigma2 * (rows - 3.0) / rows;
    let lo = 0.4_f64;
    let hi = (2.0 * v - lo * lo).sqrt();
    let levels = [-hi, -lo, lo, hi];
    let mut g = Grid { a: Vec::new(), m: Vec::new(), y: Vec::new(), x: Vec::new() };
    for a in [0.0, 1.0] {
        for x in XS {
            for e in levels {
                let m = s.alpha * a + s.gamma * x + e;
                let y = s.b0
                    + s.b1 * a
                    + s.c1 * m
                    + s.c2 * m * m
                    + s.c3 * m * m * m
                    + s.delta * a * m
                    + s.bx * x;
                g.a.push(a);
                g.m.push(m);
                g.y.push(y);
                g.x.push(x);
            }
        }
    }
    g
}

fn m1(s: &Scm, ap: f64) -> f64 {
    s.alpha * ap + s.gamma * EX
}

fn m2(s: &Scm, ap: f64) -> f64 {
    (s.alpha * ap).powi(2) + 2.0 * s.alpha * ap * s.gamma * EX + s.gamma * s.gamma * EX2 + s.sigma2
}

fn m3(s: &Scm, ap: f64) -> f64 {
    let am = s.alpha * ap;
    let e_mu3 = am.powi(3)
        + 3.0 * am * am * s.gamma * EX
        + 3.0 * am * s.gamma * s.gamma * EX2
        + s.gamma.powi(3) * EX3;
    e_mu3 + 3.0 * s.sigma2 * m1(s, ap)
}

fn theta(s: &Scm, a: f64, ap: f64) -> f64 {
    s.b0 + s.b1 * a
        + s.bx * EX
        + (s.c1 + s.delta * a) * m1(s, ap)
        + s.c2 * m2(s, ap)
        + s.c3 * m3(s, ap)
}

/// `(NDE, NIE, TE)` written out from the SCM, never from the estimator.
fn truth(s: &Scm) -> (f64, f64, f64) {
    let nde = s.b1 + s.delta * s.gamma * EX;
    let nie = (s.c1 + s.delta) * s.alpha
        + s.c2 * (s.alpha * s.alpha + 2.0 * s.alpha * s.gamma * EX)
        + s.c3 * (m3(s, 1.0) - m3(s, 0.0));
    let te = theta(s, 1.0, 1.0) - theta(s, 0.0, 0.0);
    (nde, nie, te)
}

fn config(nodes: usize) -> NonlinearMediationConfig {
    NonlinearMediationConfig {
        quadrature_nodes: nodes,
        bootstrap_replicates: 0,
        ..NonlinearMediationConfig::default()
    }
}

fn run(
    g: &Grid,
    premises: &NonlinearMediationPremises,
    cfg: &NonlinearMediationConfig,
) -> Result<NonlinearMediationEstimate, EstimationError> {
    let covs = [g.x.as_slice()];
    let input = NonlinearMediationInput {
        treatment: &g.a,
        mediator: &g.m,
        outcome: &g.y,
        covariates: &covs,
    };
    estimate_nonlinear_mediation(&input, premises, cfg)
}

fn refusal_message(r: Result<NonlinearMediationEstimate, EstimationError>) -> String {
    match r {
        Err(EstimationError::Refused { message, .. }) => message,
        other => panic!("expected a refusal, got {other:?}"),
    }
}

#[test]
fn b4_mediation_closed_form_algebra_is_self_consistent() {
    // The two ways of writing the truth (decomposed and telescoped) agree.
    let s = base_scm();
    let (nde, nie, te) = truth(&s);
    assert!((nde + nie - te).abs() < 1e-12, "closed-form TE != NDE + NIE");
    assert!((nde - (theta(&s, 1.0, 0.0) - theta(&s, 0.0, 0.0))).abs() < 1e-12);
    assert!((nie - (theta(&s, 1.0, 1.0) - theta(&s, 1.0, 0.0))).abs() < 1e-12);
}

#[test]
fn b4_mediation_exact_scm_recovers_closed_form_effects() {
    let s = base_scm();
    let g = grid(&s);
    let est = run(&g, &NonlinearMediationPremises::sequentially_ignorable(), &config(8))
        .expect("estimate");
    let (nde, nie, te) = truth(&s);
    assert!((est.natural_direct - nde).abs() < 1e-8, "NDE {} vs {nde}", est.natural_direct);
    assert!((est.natural_indirect - nie).abs() < 1e-8, "NIE {} vs {nie}", est.natural_indirect);
    assert!((est.total - te).abs() < 1e-8, "TE {} vs {te}", est.total);
    assert!((est.mediator.residual_variance - s.sigma2).abs() < 1e-9);
    assert!((est.mediator.coefficients[1] - s.alpha).abs() < 1e-9);
    assert!(est.integration_error < 1e-10);
    assert_eq!(est.calibration, "unmeasured");
    assert_eq!(est.interval_status, "closed_calibration_unmeasured");
    assert!(est.natural_direct_se.is_none(), "no replicates were requested");
    assert_eq!(est.n_rows, 40);
}

#[test]
fn b4_mediation_quadrature_error_check_reports_and_refuses() {
    // Cubic outcome: E[M^3 | a'] = mu^3 + 3*mu*sigma2 depends on mu, so the sigma2 integration
    // term does not cancel from the contrasts and the node count matters.
    let s = Scm { c3: 0.2, ..base_scm() };
    let g = grid(&s);
    let ignorable = NonlinearMediationPremises::sequentially_ignorable();
    let cubic = |nodes: usize| NonlinearMediationConfig { outcome_degree: 3, ..config(nodes) };
    // 8 versus 16 nodes integrate a cubic exactly: the reported error is rounding only.
    let fine = run(&g, &ignorable, &cubic(8)).expect("fine");
    assert!(fine.integration_error < 1e-10, "error {}", fine.integration_error);
    let (nde, nie, te) = truth(&s);
    assert!((fine.natural_direct - nde).abs() < 1e-8);
    assert!((fine.natural_indirect - nie).abs() < 1e-8);
    assert!((fine.total - te).abs() < 1e-8);
    // 1 node is the plug-in mean (drops the 3*mu*sigma2 term); 2 nodes are exact for a cubic,
    // so the n versus 2n disagreement is sigma2-sized and is refused at the 1e-6 tolerance.
    let msg = refusal_message(run(&g, &ignorable, &cubic(1)));
    assert!(msg.contains("nonlinear_mediation.integration_error"), "{msg}");
    // With a loose tolerance the estimate is returned at 2 nodes with the error reported,
    // and it still equals the closed form because 2 nodes are exact for degree 3.
    let loose = NonlinearMediationConfig { integration_tolerance: 10.0, ..cubic(1) };
    let est = run(&g, &ignorable, &loose).expect("loose");
    assert!(est.integration_error > 1e-3, "error {}", est.integration_error);
    assert!((est.natural_indirect - nie).abs() < 1e-8);
}

#[test]
fn b4_mediation_confounding_and_cross_world_premises_are_refused() {
    let g = grid(&base_scm());
    let cfg = config(8);
    let mut flags = [
        NonlinearMediationPremises {
            unmeasured_treatment_outcome_confounding: true,
            ..NonlinearMediationPremises::sequentially_ignorable()
        },
        NonlinearMediationPremises {
            unmeasured_treatment_mediator_confounding: true,
            ..NonlinearMediationPremises::sequentially_ignorable()
        },
        NonlinearMediationPremises {
            unmeasured_mediator_outcome_confounding: true,
            ..NonlinearMediationPremises::sequentially_ignorable()
        },
    ];
    for premises in &mut flags {
        let msg = refusal_message(run(&g, premises, &cfg));
        assert!(msg.contains("nonlinear_mediation.confounding"), "{msg}");
    }
    let induced = NonlinearMediationPremises {
        treatment_induced_mediator_outcome_confounders: vec!["L".to_string()],
        ..NonlinearMediationPremises::sequentially_ignorable()
    };
    let msg = refusal_message(run(&g, &induced, &cfg));
    assert!(msg.contains("nonlinear_mediation.treatment_induced_confounding"), "{msg}");
    let no_cross_world = NonlinearMediationPremises {
        cross_world_independence: false,
        ..NonlinearMediationPremises::sequentially_ignorable()
    };
    let msg = refusal_message(run(&g, &no_cross_world, &cfg));
    assert!(msg.contains("nonlinear_mediation.cross_world_independence_not_declared"), "{msg}");
    let interventional = NonlinearMediationConfig {
        estimand: NonlinearMediationEstimand::InterventionalEffects,
        ..cfg
    };
    let msg = refusal_message(run(
        &g,
        &NonlinearMediationPremises::sequentially_ignorable(),
        &interventional,
    ));
    assert!(msg.contains("nonlinear_mediation.interventional_effects_closed"), "{msg}");
}

#[test]
fn b4_mediation_total_equals_direct_plus_indirect() {
    let s = Scm { c2: -0.35, delta: -0.5, alpha: 0.4, b1: -0.2, ..base_scm() };
    let g = grid(&s);
    let est = run(&g, &NonlinearMediationPremises::sequentially_ignorable(), &config(10))
        .expect("estimate");
    assert!((est.total - (est.natural_direct + est.natural_indirect)).abs() < 1e-10);
    assert!(est.identity_residual < 1e-10);
    let (nde, nie, te) = truth(&s);
    assert!((est.natural_direct - nde).abs() < 1e-8);
    assert!((est.natural_indirect - nie).abs() < 1e-8);
    assert!((est.total - te).abs() < 1e-8);
}

#[test]
fn b4_mediation_weak_overlap_is_refused() {
    // alpha = 6: the control mediator law sits far outside the treated arm's mediator range.
    let s = Scm { alpha: 6.0, ..base_scm() };
    let g = grid(&s);
    let msg =
        refusal_message(run(&g, &NonlinearMediationPremises::sequentially_ignorable(), &config(8)));
    assert!(msg.contains("nonlinear_mediation.overlap"), "{msg}");
    // An unpopulated treated arm is the same refusal on counts.
    let full = grid(&base_scm());
    let keep: Vec<usize> = {
        let mut treated_seen = 0;
        (0..full.a.len())
            .filter(|&i| {
                if full.a[i] > 0.5 {
                    treated_seen += 1;
                    treated_seen <= 3
                } else {
                    true
                }
            })
            .collect()
    };
    let small = Grid {
        a: keep.iter().map(|&i| full.a[i]).collect(),
        m: keep.iter().map(|&i| full.m[i]).collect(),
        y: keep.iter().map(|&i| full.y[i]).collect(),
        x: keep.iter().map(|&i| full.x[i]).collect(),
    };
    let msg = refusal_message(run(
        &small,
        &NonlinearMediationPremises::sequentially_ignorable(),
        &config(8),
    ));
    assert!(msg.contains("nonlinear_mediation.overlap"), "{msg}");
}

#[test]
fn b4_mediation_linear_g_matches_product_of_coefficients_cross_check() {
    // Regression / cross-check, NOT independent truth: with g linear and no interaction the
    // natural indirect effect is the linear product of coefficients (alpha_hat * c1_hat) and the
    // natural direct effect is the treatment coefficient. The coefficients are read from the
    // exact grid by regression-free contrasts (balanced design, exact outcome fit).
    let s = Scm { c2: 0.0, delta: 0.0, ..base_scm() };
    let g = grid(&s);
    let mean_m = |arm: f64| {
        let v: Vec<f64> =
            g.a.iter()
                .zip(&g.m)
                .filter(|(a, _)| (**a - arm).abs() < 0.5)
                .map(|(_, m)| *m)
                .collect();
        v.iter().sum::<f64>() / v.len() as f64
    };
    let alpha_hat = mean_m(1.0) - mean_m(0.0);
    // Two rows in the same (A, X) cell with different M identify c1 (outcome is exact).
    let c1_hat = (g.y[1] - g.y[0]) / (g.m[1] - g.m[0]);
    let degree_one = NonlinearMediationConfig { outcome_degree: 1, ..config(4) };
    let est = run(&g, &NonlinearMediationPremises::sequentially_ignorable(), &degree_one)
        .expect("estimate");
    assert!((est.natural_indirect - alpha_hat * c1_hat).abs() < 1e-8);
    assert!((est.natural_direct - s.b1).abs() < 1e-8);
    assert!((est.total - (s.b1 + alpha_hat * c1_hat)).abs() < 1e-8);
}

#[test]
fn b4_mediation_bootstrap_is_deterministic_and_closed_for_intervals() {
    let g = grid(&base_scm());
    let ignorable = NonlinearMediationPremises::sequentially_ignorable();
    let cfg = NonlinearMediationConfig { bootstrap_replicates: 30, seed: 7, ..config(8) };
    let a = run(&g, &ignorable, &cfg).expect("first");
    let b = run(&g, &ignorable, &cfg).expect("second");
    assert_eq!(a, b, "same seed must reproduce the whole estimate");
    let se = a.natural_direct_se.expect("direct se");
    assert!(se > 0.0 && se.is_finite());
    assert!(a.natural_indirect_se.is_some() && a.total_se.is_some());
    assert_eq!(
        u64::from(a.bootstrap.replicates_succeeded) + a.bootstrap.failed_replicate_ids.len() as u64,
        30
    );
    let other = NonlinearMediationConfig { seed: 8, ..cfg };
    let c = run(&g, &ignorable, &other).expect("other seed");
    assert_ne!(a.natural_direct_se, c.natural_direct_se);
    assert_eq!(a.calibration, "unmeasured");
    assert_eq!(a.interval_status, "closed_calibration_unmeasured");
}
